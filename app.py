import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))
F1 = os.path.join(BASE, "Metadata Pos Hujan BMKGSoft v1.xlsx")
F2 = os.path.join(BASE, "Metadata v2.xlsx")

ORDER = [
    "ID", "Nama", "Latitude", "Longitude", "Elevasi (m)", "Kabupaten", "Kabupaten ID",
    "Provinsi", "Provinsi ID", "Region", "Valid From", "Valid To", "Aktif",
    "Entry Station ID", "Entry Station",
]
NUM_COLS = {"Latitude", "Longitude", "Elevasi (m)", "Kabupaten ID", "Provinsi ID",
            "Region", "Entry Station ID"}
COMPARE_COLS = [c for c in ORDER if c != "ID"]
ROMAN = {"I": "1", "II": "2", "III": "3", "IV": "4", "V": "5"}

V1_COLS = {
    "POS_HUJAN_ID": "ID", "NAME": "Nama", "CURRENT_LATITUDE": "Latitude",
    "CURRENT_LONGITUDE": "Longitude", "CURRENT_ELEVATION_M": "Elevasi (m)",
    "KABUPATEN_NAME": "Kabupaten", "KABUPATEN_ID": "Kabupaten ID",
    "PROPINSI_NAME": "Provinsi", "PROPINSI_ID": "Provinsi ID", "REGION_ID": "Region",
    "VALID_FROM": "Valid From", "VALID_TO": "Valid To", "IS_ACTIVE": "Aktif",
    "ENTRY_STATION_ID": "Entry Station ID", "ENTRY_STATION_NAME": "Entry Station",
}
V2_COLS = {
    "pos_hujan_id": "ID", "pos_hujan_name": "Nama", "latitude": "Latitude",
    "longitude": "Longitude", "elevation_m": "Elevasi (m)",
    "kabupaten_name": "Kabupaten", "kabupaten_id": "Kabupaten ID",
    "provinsi_name": "Provinsi", "provinsi_id": "Provinsi ID", "region_desc": "Region",
    "valid_from": "Valid From", "valid_to": "Valid To", "is_active": "Aktif",
    "entry_bmkgsoft_id": "Entry Station ID", "entry_bmkgsoft_name": "Entry Station",
}


def norm_text(v):
    if v is None or pd.isna(v):
        return ""
    return str(v).strip()


def norm_active(v):
    s = norm_text(v).lower()
    if s in ("y", "true", "yes", "1"):
        return True
    if s in ("n", "false", "no", "0"):
        return False
    return None


def norm_region(v):
    s = str(v).strip()
    if s.lower().startswith("region"):
        tail = s.split()[-1].strip().upper()
        return int(ROMAN[tail]) if tail in ROMAN else tail
    try:
        f = float(s)
        return int(f) if f.is_integer() else f
    except ValueError:
        return s


def norm_date(v):
    if v is None or pd.isna(v):
        return None
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.strftime("%Y-%m-%d")
    s = str(v).strip()
    if s in ("", "NaT", "nan", "None"):
        return None
    d = pd.to_datetime(s, errors="coerce")
    return None if pd.isna(d) else d.strftime("%Y-%m-%d")


@st.cache_data(show_spinner="Membaca file Excel...")
def load_data():
    raw1 = pd.read_excel(F1, sheet_name=0).rename(columns=V1_COLS)
    raw2 = pd.read_excel(F2, sheet_name="v21").rename(columns=V2_COLS)
    frames = []
    for raw in (raw1, raw2):
        df = raw[ORDER].dropna(how="all").copy()
        df["ID"] = df["ID"].map(norm_text)
        for c in ("Nama", "Kabupaten", "Provinsi", "Entry Station"):
            df[c] = df[c].map(norm_text)
        df["Aktif"] = df["Aktif"].map(norm_active)
        df["Region"] = df["Region"].map(norm_region)
        for c in ("Valid From", "Valid To"):
            df[c] = df[c].map(norm_date)
        for c in NUM_COLS - {"Region"}:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        frames.append(df.reset_index(drop=True))
    return frames[0], frames[1]


@st.cache_data
def load_notes():
    return pd.read_excel(F2, sheet_name="Sheet1", header=None).dropna(how="all")


def format_series(s, col):
    if col in NUM_COLS:
        def f(v):
            if pd.isna(v):
                return ""
            f = float(v)
            return str(int(f)) if f.is_integer() else str(f)
        return s.map(f).to_numpy()
    if col == "Aktif":
        return s.map(lambda v: "" if pd.isna(v) else ("Ya" if bool(v) else "Tidak")).to_numpy()
    return s.map(norm_text).replace("", "—").to_numpy()


@st.cache_data
def compute_diff(df1, df2):
    d1 = df1.drop_duplicates("ID").set_index("ID")
    d2 = df2.drop_duplicates("ID").set_index("ID")
    common = d1.index.intersection(d2.index)
    only1 = df1[~df1["ID"].isin(d2.index)].sort_values("ID").reset_index(drop=True)
    only2 = df2[~df2["ID"].isin(d1.index)].sort_values("ID").reset_index(drop=True)

    a = d1.loc[common]
    b = d2.loc[common]
    parts = []
    for col in COMPARE_COLS:
        va, vb = a[col], b[col]
        if col in NUM_COLS:
            na = pd.to_numeric(va, errors="coerce").to_numpy(dtype=float)
            nb = pd.to_numeric(vb, errors="coerce").to_numpy(dtype=float)
            bad = ~np.isclose(na, nb, rtol=0.0, atol=1e-9, equal_nan=True)
        elif col == "Provinsi":
            bad = ~np.isclose(pd.to_numeric(a["Provinsi ID"], errors="coerce").to_numpy(dtype=float),
                              pd.to_numeric(b["Provinsi ID"], errors="coerce").to_numpy(dtype=float),
                              rtol=0.0, atol=1e-9, equal_nan=True)
        elif col == "Aktif":
            bad = ~(va.to_numpy(dtype=object) == vb.to_numpy(dtype=object))
        else:
            bad = va.map(norm_text).to_numpy() != vb.map(norm_text).to_numpy()
        if bad.any():
            parts.append(pd.DataFrame({
                "ID": common.to_numpy()[bad],
                "Nama": a["Nama"].to_numpy()[bad],
                "Provinsi": a["Provinsi"].to_numpy()[bad],
                "Kabupaten": a["Kabupaten"].to_numpy()[bad],
                "Kolom": col,
                "Nilai V1": format_series(va, col)[bad],
                "Nilai V2": format_series(vb, col)[bad],
            }))
    cols = ["ID", "Nama", "Provinsi", "Kabupaten", "Kolom", "Nilai V1", "Nilai V2"]
    diff = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
    return only1, only2, diff, set(common)


@st.cache_data
def build_side(df1, df2):
    a = df1.drop_duplicates("ID").set_index("ID").copy()
    b = df2.drop_duplicates("ID").set_index("ID").copy()
    common = a.index.intersection(b.index)
    a = a.loc[common]
    b = b.loc[common]
    mask = pd.DataFrame(False, index=common, columns=COMPARE_COLS)
    for col in COMPARE_COLS:
        va, vb = a[col], b[col]
        if col in NUM_COLS:
            na = pd.to_numeric(va, errors="coerce").to_numpy(dtype=float)
            nb = pd.to_numeric(vb, errors="coerce").to_numpy(dtype=float)
            mask[col] = ~np.isclose(na, nb, rtol=0.0, atol=1e-9, equal_nan=True)
        elif col == "Provinsi":
            mask[col] = ~np.isclose(pd.to_numeric(a["Provinsi ID"], errors="coerce").to_numpy(dtype=float),
                                    pd.to_numeric(b["Provinsi ID"], errors="coerce").to_numpy(dtype=float),
                                    rtol=0.0, atol=1e-9, equal_nan=True)
        elif col == "Aktif":
            mask[col] = ~(va.to_numpy(dtype=object) == vb.to_numpy(dtype=object))
        else:
            mask[col] = va.map(norm_text).to_numpy() != vb.map(norm_text).to_numpy()
    disp_cols = ["Jumlah Selisih"] + COMPARE_COLS
    n_beda = mask.sum(axis=1).astype(int)
    a.insert(0, "Jumlah Selisih", n_beda)
    b.insert(0, "Jumlah Selisih", n_beda)
    mask.insert(0, "Jumlah Selisih", False)
    a = a[disp_cols]
    b = b[disp_cols]
    for df_ in (a, b):
        df_["Aktif"] = df_["Aktif"].map(lambda v: "" if pd.isna(v) else ("Ya" if v else "Tidak"))
        for c in ("Valid From", "Valid To"):
            df_[c] = df_[c].map(lambda v: "" if v is None or pd.isna(v) else v)
    for col in ("Elevasi (m)", "Kabupaten ID", "Provinsi ID", "Region", "Entry Station ID"):
        a[col] = a[col].map(lambda v: "" if pd.isna(v) else (str(int(v)) if float(v).is_integer() else str(v)))
        b[col] = b[col].map(lambda v: "" if pd.isna(v) else (str(int(v)) if float(v).is_integer() else str(v)))
    return a, b, mask


def side_styler(disp, msel):
    def warna(s):
        m = msel.loc[s.index, s.name]
        return ["background-color:#f8d7da" if f else "" for f in m]
    return disp.style.apply(warna, axis=0)


def to_csv_bytes(df):
    return df.to_csv(index=False, sep=";").encode("utf-8-sig")


def filter_frame(df, key_prefix, prov_col="Provinsi"):
    c1, c2 = st.columns(2)
    prov = c1.selectbox("Provinsi", ["(semua)"] + sorted(df[prov_col].unique()),
                        key=f"{key_prefix}_prov")
    q = c2.text_input("Cari (ID atau nama)", "", key=f"{key_prefix}_cari").strip().lower()
    out = df
    if prov != "(semua)":
        out = out[out[prov_col] == prov]
    if q:
        out = out[out["ID"].str.lower().str.contains(q, na=False)
                  | out["Nama"].str.lower().str.contains(q, na=False)]
    return out.reset_index(drop=True)


def tampilkan_hanya(df, nama_file, kosong):
    st.caption(f"Baris yang hanya ada di {nama_file}, tidak ada di file pasangannya.")
    out = filter_frame(df, nama_file)
    if out.empty:
        st.info(kosong)
    else:
        st.dataframe(out, use_container_width=True, height=550, hide_index=True)
        st.caption(f"{len(out):,} baris".replace(",", "."))
    st.download_button("Unduh CSV", to_csv_bytes(out), nama_file.lower().replace(" ", "_") + ".csv",
                       "text/csv", use_container_width=True, key=f"dl_{nama_file}")


def main():
    st.set_page_config(page_title="Perbandingan Metadata Pos Hujan V1 vs V2", layout="wide")
    st.title("Perbandingan Metadata Pos Hujan — V1 vs V2")
    st.caption("V1: `Metadata Pos Hujan BMKGSoft v1.xlsx` (sheet utama)  •  "
               "V2: `Metadata v2.xlsx` (sheet `v21`)")

    df1, df2 = load_data()
    only1, only2, diff, common_ids = compute_diff(df1, df2)
    a_side, b_side, mask_side = build_side(df1, df2)

    with st.sidebar:
        st.header("Sumber data")
        st.markdown(f"**V1** — `{os.path.basename(F1)}`  \n{len(df1):,} baris".replace(",", "."))
        st.markdown(f"**V2** — `{os.path.basename(F2)}` (sheet `v21`)  \n{len(df2):,} baris".replace(",", "."))
        st.divider()
        st.caption(
            "Perbandingan memakai `pos_hujan_id` sebagai kunci.\n\n"
            "Format yang dinormalisasi agar bisa dibandingkan:\n"
            "- IS_ACTIVE: `Y`/`N` (V1) vs `true`/`false` (V2)\n"
            "- Region: angka (V1) vs `Region I...V` (V2)\n"
            "- Tanggal: datetime (V1) vs teks (V2)\n"
            "- Nama provinsi lama vs baru (mis. `Nanggroe Aceh Darussalam` vs `Aceh`) "
            "dianggap sama selama Provinsi ID sama\n"
            "- Spasi dan kapitalisasi teks"
        )

    tab_ring, tab_only1, tab_only2, tab_diff, tab_side, tab_raw = st.tabs(
        ["Ringkasan", "Hanya di V1", "Hanya di V2", "Selisih Data", "Kiri–Kanan", "Data Mentah"])

    with tab_ring:
        n_pos = diff["ID"].nunique() if not diff.empty else 0
        m = st.columns(5)
        m[0].metric("Total V1", f"{len(df1):,}".replace(",", "."))
        m[1].metric("Total V2", f"{len(df2):,}".replace(",", "."))
        m[2].metric("ID sama", f"{len(common_ids):,}".replace(",", "."))
        m[3].metric("Hanya di V1", len(only1))
        m[4].metric("Hanya di V2", len(only2))
        m2 = st.columns(2)
        m2[0].metric("Pos dengan selisih", f"{n_pos:,}".replace(",", "."))
        m2[1].metric("Total selisih (sel)", f"{len(diff):,}".replace(",", "."))

        st.subheader("Jumlah selisih per kolom")
        left, right = st.columns([1, 2])
        if diff.empty:
            st.success("Tidak ada selisih antara V1 dan V2.")
        else:
            summary = diff["Kolom"].value_counts().rename_axis("Kolom").reset_index(name="Jumlah")
            with left:
                st.dataframe(summary, use_container_width=True, hide_index=True)
            with right:
                st.bar_chart(summary.set_index("Kolom")["Jumlah"], use_container_width=True)
            st.download_button("Unduh semua selisih (CSV)", to_csv_bytes(diff),
                               "selisih_v1_v2.csv", "text/csv", key="dl_ringkasan")
        st.divider()
        dup1 = int(df1["ID"].duplicated().sum())
        dup2 = int(df2["ID"].duplicated().sum())
        st.markdown(
            f"**Kualitas data:** duplikat ID di V1: {dup1} • di V2: {dup2}. "
            f"Valid To terisi di V1: {int(df1['Valid From'].notna().sum() and df1['Valid To'].notna().sum())} baris, "
            f"di V2: {int(df2['Valid To'].notna().sum())} baris."
        )
        with st.expander("Catatan parameter V2 (sheet `Sheet1` di file v2)"):
            st.dataframe(load_notes(), use_container_width=True, hide_index=True)

    with tab_only1:
        tampilkan_hanya(only1, "Hanya V1", "Tidak ada pos yang hanya ada di V1 — semua ID V1 juga ada di V2.")

    with tab_only2:
        tampilkan_hanya(only2, "Hanya V2", "Tidak ada pos yang hanya ada di V2 — semua ID V2 juga ada di V1.")

    with tab_diff:
        st.caption(f"{n_pos:,} pos dengan selisih dari {len(common_ids):,} pos yang sama — "
                   f"total {len(diff):,} sel berbeda. Biarkan kolom kosong untuk menampilkan semua kolom.".replace(",", "."))
        c1, c2, c3 = st.columns([2, 3, 2])
        prov = c1.selectbox("Provinsi (menurut V1)",
                            ["(semua)"] + sorted(diff["Provinsi"].unique()) if not diff.empty else ["(semua)"],
                            key="diff_prov")
        kolom = c2.multiselect("Kolom", COMPARE_COLS, default=[], key="diff_kolom")
        q = c3.text_input("Cari (ID atau nama)", "", key="diff_cari").strip().lower()

        disp = diff
        if prov != "(semua)":
            disp = disp[disp["Provinsi"] == prov]
        if kolom:
            disp = disp[disp["Kolom"].isin(kolom)]
        if q:
            disp = disp[disp["ID"].str.lower().str.contains(q, na=False)
                        | disp["Nama"].str.lower().str.contains(q, na=False)]
        disp = disp.reset_index(drop=True)

        st.bar_chart(disp["Kolom"].value_counts(), use_container_width=True)
        st.dataframe(disp, use_container_width=True, height=550, hide_index=True)
        st.caption(f"{len(disp):,} baris selisih".replace(",", "."))
        st.download_button("Unduh hasil filter (CSV)", to_csv_bytes(disp),
                           "selisih_terfilter.csv", "text/csv", key="dl_diff")

    with tab_side:
        st.caption("Satu tabel: setiap kolom V1 bersebelahan dengan pasangan V2-nya — "
                   "sel berlatar merah berarti nilainya berbeda antara V1 dan V2.")
        c1, c2, c3 = st.columns(3)
        hanya_beda = c1.checkbox("Hanya baris dengan selisih", True, key="side_beda")
        prov = c2.selectbox("Provinsi (menurut V1)", ["(semua)"] + sorted(a_side["Provinsi"].unique()),
                            key="side_prov")
        q = c3.text_input("Cari (ID atau nama)", "", key="side_cari").strip().lower()
        pilih = st.pills("Kolom yang dicek perbedaannya", COMPARE_COLS,
                         selection_mode="multi", default=COMPARE_COLS, key="side_pills",
                         help="Kolom yang tidak dipilih tetap tampil, tetapi tidak diwarnai merah "
                              "dan tidak dihitung sebagai selisih.")
        aktif = set(pilih)

        jumlah_cek = (mask_side[list(aktif)].sum(axis=1).astype(int)
                      if aktif else pd.Series(0, index=mask_side.index))
        sel = a_side
        if hanya_beda:
            sel = sel[jumlah_cek.loc[sel.index] > 0]
        if prov != "(semua)":
            sel = sel[sel["Provinsi"] == prov]
        if q:
            sel = sel[sel.index.str.lower().str.contains(q, na=False)
                      | sel["Nama"].str.lower().str.contains(q, na=False)]
        total_terfilter = len(sel)
        nmax = st.slider("Maksimal baris ditampilkan", 50, max(50, total_terfilter),
                         max(50, total_terfilter), step=50,
                         help="Default: semua baris yang lolos filter. "
                              "Semakin banyak baris, semakin lama render tabel.")
        idx = sel.index[:nmax]

        if idx.empty:
            st.info("Tidak ada baris yang sesuai filter.")
        else:
            disp = pd.DataFrame(index=idx)
            disp["Jumlah Selisih"] = jumlah_cek.loc[idx]
            msel = pd.DataFrame(False, index=idx, columns=["Jumlah Selisih"])
            for c in COMPARE_COLS:
                disp[f"{c} (V1)"] = a_side.loc[idx, c]
                disp[f"{c} (V2)"] = b_side.loc[idx, c]
                msk = mask_side.loc[idx, c] if c in aktif else pd.Series(False, index=idx)
                msel[f"{c} (V1)"] = msk
                msel[f"{c} (V2)"] = msk
            st.dataframe(side_styler(disp, msel), use_container_width=True, height=650)
            if total_terfilter > nmax:
                st.caption(f"Menampilkan {nmax:,} dari {total_terfilter:,} baris terfilter — "
                           "perbesar lewat slider di atas.".replace(",", "."))
            else:
                st.caption(f"{total_terfilter:,} baris terfilter.".replace(",", "."))
            st.download_button("Unduh tampilan ini (CSV)", to_csv_bytes(disp.reset_index()),
                               "v1_v2_bersebelahan.csv", "text/csv", key="dl_side")

    with tab_raw:
        versi = st.radio("Tampilkan file", ["V1", "V2"], horizontal=True)
        src = df1 if versi == "V1" else df2
        tampil = filter_frame(src, f"raw_{versi}")
        tampil = tampil.copy()
        tampil["Aktif"] = tampil["Aktif"].map(lambda v: "" if pd.isna(v) else ("Ya" if v else "Tidak"))
        st.dataframe(tampil, use_container_width=True, height=550, hide_index=True)
        st.caption(f"{len(tampil):,} baris".replace(",", "."))
        st.download_button("Unduh CSV", to_csv_bytes(tampil), f"data_v{versi[-1]}.csv",
                           "text/csv", key=f"dl_raw_{versi}")


if __name__ == "__main__":
    main()
