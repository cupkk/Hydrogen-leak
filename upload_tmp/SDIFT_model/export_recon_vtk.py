import argparse
import os

import numpy as np
import scipy.io as sio


def write_array(f, arr, per_line=6, indent=""):
    for i in range(0, len(arr), per_line):
        line = " ".join(f"{v:.6e}" for v in arr[i:i+per_line])
        f.write(f"{indent}{line}\n")


def write_rectilinear_vtk(path, data, x, y, z, field_name="concentration"):
    nx, ny, nz = data.shape
    with open(path, "w", encoding="ascii") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"{field_name}\n")
        f.write("ASCII\n")
        f.write("DATASET RECTILINEAR_GRID\n")
        f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        f.write(f"X_COORDINATES {nx} float\n")
        write_array(f, x.astype(np.float32))
        f.write(f"Y_COORDINATES {ny} float\n")
        write_array(f, y.astype(np.float32))
        f.write(f"Z_COORDINATES {nz} float\n")
        write_array(f, z.astype(np.float32))
        f.write(f"POINT_DATA {nx * ny * nz}\n")
        f.write(f"SCALARS {field_name} float 1\n")
        f.write("LOOKUP_TABLE default\n")
        write_array(f, data.ravel(order="F").astype(np.float32))


def write_rectilinear_vtr(path, data, x, y, z, field_name="concentration"):
    nx, ny, nz = data.shape
    extent = f"0 {nx - 1} 0 {ny - 1} 0 {nz - 1}"
    with open(path, "w", encoding="ascii") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="RectilinearGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write(f'  <RectilinearGrid WholeExtent="{extent}">\n')
        f.write(f'    <Piece Extent="{extent}">\n')
        f.write(f'      <PointData Scalars="{field_name}">\n')
        f.write(f'        <DataArray type="Float32" Name="{field_name}" NumberOfComponents="1" format="ascii">\n')
        write_array(f, data.ravel(order="F").astype(np.float32), indent="          ")
        f.write("        </DataArray>\n")
        f.write("      </PointData>\n")
        f.write("      <CellData>\n")
        f.write("      </CellData>\n")
        f.write("      <Coordinates>\n")
        f.write('        <DataArray type="Float32" Name="X_COORDINATES" NumberOfComponents="1" format="ascii">\n')
        write_array(f, x.astype(np.float32), indent="          ")
        f.write("        </DataArray>\n")
        f.write('        <DataArray type="Float32" Name="Y_COORDINATES" NumberOfComponents="1" format="ascii">\n')
        write_array(f, y.astype(np.float32), indent="          ")
        f.write("        </DataArray>\n")
        f.write('        <DataArray type="Float32" Name="Z_COORDINATES" NumberOfComponents="1" format="ascii">\n')
        write_array(f, z.astype(np.float32), indent="          ")
        f.write("        </DataArray>\n")
        f.write("      </Coordinates>\n")
        f.write("    </Piece>\n")
        f.write("  </RectilinearGrid>\n")
        f.write("</VTKFile>\n")


def write_pvd(path, files, times):
    with open(path, "w", encoding="ascii") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        f.write("  <Collection>\n")
        for fname, t in zip(files, times):
            f.write(f'    <DataSet timestep="{t:.6f}" group="" part="0" file="{fname}"/>\n')
        f.write("  </Collection>\n")
        f.write("</VTKFile>\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon_mat", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--out_dir", default="vtk_out")
    parser.add_argument("--field_name", default="concentration")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--time_start", type=int, default=0)
    parser.add_argument("--time_end", type=int, default=-1)
    parser.add_argument("--time_stride", type=int, default=1)
    parser.add_argument("--use_real_coords", action="store_true", default=True)
    parser.add_argument("--recon_key", default="recon_list")
    parser.add_argument("--prefix", default="recon")
    parser.add_argument("--write_pvd", action="store_true", default=True)
    parser.add_argument("--vtk_format", choices=["xml", "legacy"], default="xml")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    d = sio.loadmat(args.recon_mat)
    if args.recon_key not in d:
        raise KeyError(f"{args.recon_key} not found in {args.recon_mat}")
    recon = d[args.recon_key]
    if args.sample_index >= recon.shape[0]:
        raise ValueError("sample_index out of range")

    meta = np.load(args.meta, allow_pickle=True).item()["data"]
    if args.use_real_coords:
        x = meta.get("u_ind_real", meta["u_ind_uni"])
        y = meta.get("v_ind_real", meta["v_ind_uni"])
        z = meta.get("w_ind_real", meta["w_ind_uni"])
        t = meta.get("t_ind_real", meta["t_ind_uni"])
    else:
        x = meta["u_ind_uni"]
        y = meta["v_ind_uni"]
        z = meta["w_ind_uni"]
        t = meta["t_ind_uni"]

    t_end = recon.shape[1] if args.time_end < 0 else min(args.time_end, recon.shape[1])
    t_indices = list(range(args.time_start, t_end, args.time_stride))
    out_files = []
    out_times = []

    for idx in t_indices:
        frame = recon[args.sample_index, idx]
        ext = ".vtr" if args.vtk_format == "xml" else ".vtk"
        fname = f"{args.prefix}_{idx:04d}{ext}"
        out_path = os.path.join(args.out_dir, fname)
        if args.vtk_format == "xml":
            write_rectilinear_vtr(out_path, frame, x, y, z, field_name=args.field_name)
        else:
            write_rectilinear_vtk(out_path, frame, x, y, z, field_name=args.field_name)
        out_files.append(fname)
        out_times.append(float(t[idx]) if idx < len(t) else float(idx))

    if args.write_pvd:
        write_pvd(os.path.join(args.out_dir, f"{args.prefix}.pvd"), out_files, out_times)


if __name__ == "__main__":
    main()
