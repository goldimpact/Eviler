import json
import os
import glob
import numpy as np
from tqdm import tqdm
from functools import wraps
from PIL import Image
from pytoshop import PsdFile
from pytoshop.user import nested_layers
from pytoshop.enums import ColorMode, BlendMode, ChannelId, ColorDepth, Version

class HierarchyNode:
    def __init__(self, name, kind_index, parent=None):
        self.name = name
        self.kind_index = kind_index
        self.parent = parent
        self.elements = []

def parse_hierarchy(file_path):
    root = HierarchyNode("ROOT", -1)
    stack = [{"node": root, "indent": -1}]
    with open(file_path, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip())
            content = line.strip()
            while stack[-1]["indent"] >= indent:
                stack.pop()
            parent_info = stack[-1]
            if ',' in content:
                name, kind = content.split(',', 1)
                node = HierarchyNode(name.strip(), int(kind.strip()), parent_info["node"])
                parent_info["node"].elements.append(node)
                stack.append({"node": node, "indent": indent})
            else:
                artmesh = int(content)
                parent_info["node"].elements.append(artmesh)
    return root

def load_part_names(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {part["Id"]: part["Name"] for part in data.get("Parts", [])}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def with_progress_bar(func):
    @wraps(func)
    def wrapper(node, mesh_mapping, part_names=None, progress=None):
        total = count_elements(node)
        if progress is None:
            with tqdm(total=total, desc="Building PSD Structure", unit="element") as pbar:
                return func(node, mesh_mapping, part_names, pbar)
        else:
            return func(node, mesh_mapping, part_names, progress)
    return wrapper

def count_elements(node):
    count = 0
    for elem in node.elements:
        count += 1
        if isinstance(elem, HierarchyNode):
            count += count_elements(elem)
    return count

@with_progress_bar
def build_layer_structure(node, mesh_mapping, part_names=None, progress=None):
    layers = []
    for elem in node.elements:
        if progress:
            progress.update(1)
            progress.set_postfix_str(f"Processing: {get_element_name(elem)}") 
        if isinstance(elem, int):
            if elem in mesh_mapping:
                layers.append(create_image_layer(mesh_mapping[elem]))
        else:
            resolved_name = part_names.get(elem.name, elem.name) if part_names else elem.name
            ret = build_layer_structure(elem, mesh_mapping, part_names, progress)
            if len(ret) == 0:
                continue
            group = create_group(resolved_name, ret)
            layers.append(group)
    return layers

def get_element_name(elem):
    if isinstance(elem, int):
        return f"Mesh#{elem}"
    elif isinstance(elem, HierarchyNode):
        return f"Group:{elem.name}"
    return "Unknown"

def create_group(name, layers):
    return nested_layers.Group(
        name=name,
        layers=layers,
        visible=True,
        opacity=255,
        blend_mode=BlendMode.normal,
        closed=False
    )

def get_sorted_png_files(png_dir):
    files = glob.glob(os.path.join(png_dir, "*_*.png"))
    file_info = []
    for path in files:
        filename = os.path.basename(path)
        try:
            order_str, mesh_str = filename.split('_', 1)
            order = int(order_str)
            mesh_index = int(os.path.splitext(mesh_str)[0])
            file_info.append( (order, mesh_index, path) )
        except ValueError:
            print(f"ValueError {filename}")
    file_info.sort(key=lambda x: x[0])
    return [(mesh, path) for _, mesh, path in file_info]

def get_bounding_box(img):
    alpha = np.array(img.getchannel('A'))
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0) 
    if not (rows.any() and cols.any()):
        return (0, 0, img.width, img.height) 
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]    
    return (xmin, ymin, xmax + 1, ymax + 1)

def create_image_layer(path):
    with Image.open(path) as img:
        bbox = get_bounding_box(img)
        xmin, ymin, xmax, ymax = bbox    
        cropped = img.crop(bbox) 
        arr = np.array(cropped.convert("RGBA")).astype(np.float32) / 255.0
        a = arr[:, :, 3]
        mask = a > 0
        for c in range(3):
            arr[:, :, c][mask] /= a[mask]    
        channels = {
            ChannelId.red: (arr[:, :, 0] * 255).astype(np.uint8),
            ChannelId.green: (arr[:, :, 1] * 255).astype(np.uint8),
            ChannelId.blue: (arr[:, :, 2] * 255).astype(np.uint8),
            ChannelId.transparency: (arr[:, :, 3] * 255).astype(np.uint8)
        }
        return nested_layers.Image(
            name=os.path.basename(path),
            visible=True,
            opacity=255,
            blend_mode=BlendMode.normal,
            top=int(ymin),
            left=int(xmin),
            bottom=int(ymax),
            right=int(xmax),
            channels=channels,
            color_mode=ColorMode.rgb
        )

def pack_pngs_to_psd(png_dir, hierarchy_file, output_psd):
    sorted_files = get_sorted_png_files(png_dir)
    hierarchy_root = parse_hierarchy(hierarchy_file)
    part_names = {}
    if os.path.basename(hierarchy_file).lower() == "part.txt":
        cdi_path = os.path.join(os.path.dirname(hierarchy_file), "input.cdi3.json")
        part_names = load_part_names(cdi_path)
    mesh_mapping = {mesh: path for mesh, path in sorted_files}
    root_layers = build_layer_structure(hierarchy_root, mesh_mapping, part_names) 
    max_width, max_height = 0, 0
    for _, path in sorted_files:
        with Image.open(path) as img:
            max_width = max(max_width, img.width)
            max_height = max(max_height, img.height)
    psd = nested_layers.nested_layers_to_psd(
        layers=root_layers,
        color_mode=ColorMode.rgb,
        depth=ColorDepth.depth8,
        size=(max_width, max_height),
        version=Version.version_1,
        compression=0
    )
    with open(output_psd, 'wb') as f:
        psd.write(f)

if __name__ == "__main__":
    filename = "output"
    pack_pngs_to_psd(
        png_dir=r"..\imgs",
        hierarchy_file=r"..\order.txt",
        output_psd=filename + ".order.psd"
    )
    pack_pngs_to_psd(
        png_dir=r"..\imgs",
        hierarchy_file=r"..\part.txt",
        output_psd=filename + ".part.psd"
    )