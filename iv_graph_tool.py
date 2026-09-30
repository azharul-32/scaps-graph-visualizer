"""
SCAPS .iv File Graph Generator - Fully Customizable Tool
============================================================
Upload SCAPS .iv file(s) and build fully customized, publication-style
graphs: pick columns, colors, line styles, fonts, overlay multiple
simulations, add custom annotations, auto-generate a figure caption,
or batch-process many files into a grid of consistently styled plots.

HOW TO RUN:
  1. Install once:  pip install streamlit pandas matplotlib numpy
  2. Run:            streamlit run iv_graph_tool.py
  3. Your browser opens automatically with the interface.
"""

import re
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, LogLocator, AutoMinorLocator
import streamlit as st

st.set_page_config(page_title="SCAPS .iv Graph Generator", layout="wide")


# ============================================================
# PARSER - reads ALL columns directly from the file's own header
# ============================================================

def parse_iv_file(file_bytes):
    text = file_bytes.decode("utf-8", errors="replace")
    lines = text.splitlines()

    simulations = []
    current_data_rows = []
    current_columns = None
    current_sim_number = None
    in_data = False
    summary = {}

    for line in lines:
        stripped = line.strip()

        if "Single shot simulation" in stripped or "Batch simulation" in stripped:
            if current_columns is not None and current_data_rows:
                df = pd.DataFrame(current_data_rows, columns=current_columns)
                simulations.append({"sim_number": current_sim_number,
                                     "data": df, "summary": summary})
            current_data_rows = []
            current_columns = None
            summary = {}
            m = re.search(r"#\s*(\d+)", stripped)
            step_m = re.search(r"step\s+(\d+)", stripped)
            if step_m:
                current_sim_number = int(step_m.group(1))
            elif m:
                current_sim_number = int(m.group(1))
            else:
                current_sim_number = len(simulations) + 1
            in_data = False
            continue

        if current_columns is None and stripped.startswith("v(V)"):
            current_columns = [c.strip() for c in re.split(r"\t+", stripped) if c.strip()]
            in_data = True
            continue

        if in_data:
            if stripped == "":
                continue
            if stripped.startswith("solar"):
                in_data = False
                continue
            parts = re.split(r"\t+|\s{2,}", stripped)
            parts = [p for p in parts if p.strip() != ""]
            try:
                row = [float(p) for p in parts]
                if len(row) == len(current_columns):
                    current_data_rows.append(row)
            except ValueError:
                continue
            continue

        m = re.match(r"([A-Za-z_]+)\s*=\s*([\d.eE+-]+)", stripped)
        if m:
            summary[m.group(1)] = float(m.group(2))

    if current_columns is not None and current_data_rows:
        df = pd.DataFrame(current_data_rows, columns=current_columns)
        simulations.append({"sim_number": current_sim_number,
                             "data": df, "summary": summary})

    return simulations


FRIENDLY_NAMES = {
    "jtot(mA/cm2)": "Total Current Density",
    "j_total_rec(mA/cm2)": "Total Recombination Current",
    "j_total_gen(mA/cm2)": "Total Generation Current",
    "jbulk(mA/cm2)": "Bulk Recombination Current",
    "jifr(mA/cm2)": "Interface Recombination Current",
    "jminor_left(mA/cm2)": "Minority Carrier Recomb. (Left Contact)",
    "jminor_right(mA/cm2)": "Minority Carrier Recomb. (Right Contact)",
    "j_SRH(mA/cm2)": "SRH Recombination Current",
    "j_Radiative(mA/cm2)": "Radiative Recombination Current",
    "j_Auger(mA/cm2)": "Auger Recombination Current",
}


def friendly(col):
    return FRIENDLY_NAMES.get(col, col)


def find_voc(df, x_col):
    jtot_col = next((c for c in df.columns if "jtot" in c), None)
    if jtot_col is None:
        return None
    x_vals, j_vals = df[x_col].values, df[jtot_col].values
    sign_changes = np.where(np.diff(np.sign(j_vals)) != 0)[0]
    if len(sign_changes) == 0:
        return None
    idx = sign_changes[0]
    x0, x1 = x_vals[idx], x_vals[idx + 1]
    j0, j1 = j_vals[idx], j_vals[idx + 1]
    return x0 + (0 - j0) * (x1 - x0) / (j1 - j0)


LINESTYLE_MAP = {"Solid": "-", "Dashed": "--", "Dotted": ":", "Dash-dot": "-."}
BRIGHT_PALETTE = ['#e60000', '#009e3d', '#ff8c00', '#8b00ff', '#00b3b3', '#d4a017',
                  '#ff1493', '#1a75ff']


def apply_axis_styling(ax, x_col, font_size, show_ticks_and_grid, log_scale,
                        y_min_val, y_max_val):
    x_data_min, x_data_max = ax.get_xlim()
    if x_data_max - x_data_min <= 2.0:
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.tick_params(axis='x', which='major', length=5, width=0.7, direction='out')
        if show_ticks_and_grid:
            ax.xaxis.set_minor_locator(MultipleLocator(0.01))
            ax.tick_params(axis='x', which='minor', length=2.5, width=0.5, direction='out')

    if log_scale:
        if show_ticks_and_grid:
            ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1,
                                                    numticks=100))
    else:
        ax.yaxis.set_major_locator(MultipleLocator(10))
        if show_ticks_and_grid:
            ax.yaxis.set_minor_locator(MultipleLocator(1))
        if y_min_val is not None and y_max_val is not None:
            ax.set_ylim(y_min_val, y_max_val)

    ax.tick_params(axis='y', which='major', length=5, width=0.7, direction='out')
    if show_ticks_and_grid:
        ax.tick_params(axis='y', which='minor', length=2.5, width=0.5, direction='out')

    if show_ticks_and_grid:
        ax.grid(True, which='major', linestyle='-', linewidth=0.35, color='#e0e0e0')
        ax.grid(True, which='minor', linestyle='-', linewidth=0.2, color='#eeeeee')
    else:
        ax.grid(False)

    for spine in ax.spines.values():
        spine.set_linewidth(0.7)

    ax.tick_params(axis='both', labelsize=font_size * 0.85)


# ============================================================
# STREAMLIT INTERFACE
# ============================================================

st.title("SCAPS .iv File Graph Generator")
st.markdown(
    "<p style='font-size:16px; margin-bottom:0px;'>By Azharul Islam</p>"
    "<p style='font-size:13px; color:gray; margin-top:2px;'>University of Chittagong</p>",
    unsafe_allow_html=True
)
st.caption("Upload SCAPS .iv file(s) and build a fully customized graph. "
           "All values are read directly from the file - no approximation.")

if "annotations" not in st.session_state:
    st.session_state.annotations = []

mode = st.radio("Mode", ["Single File (full customization)",
                          "Batch Mode (multiple files, grid of graphs)"],
                 horizontal=True)

# ============================================================
# BATCH MODE
# ============================================================

if mode.startswith("Batch"):
    st.subheader("Batch Mode")
    uploaded_files = st.file_uploader("Upload multiple .iv files", type=["iv", "txt"],
                                       accept_multiple_files=True)

    if uploaded_files:
        all_parsed = []
        for uf in uploaded_files:
            sims = parse_iv_file(uf.read())
            if sims:
                all_parsed.append({"name": uf.name, "sim": sims[0]})

        if all_parsed:
            common_cols = set(all_parsed[0]["sim"]["data"].columns)
            for p in all_parsed[1:]:
                common_cols &= set(p["sim"]["data"].columns)
            common_cols = sorted(common_cols)

            x_col = st.selectbox("X-axis parameter (applied to all files)", common_cols,
                                  index=0)
            y_col = st.selectbox("Y-axis parameter (applied to all files)",
                                  [c for c in common_cols if c != x_col],
                                  format_func=friendly)
            batch_color = st.color_picker("Curve color (applied to all)", "#e60000")

            if st.button("Generate Grid", type="primary"):
                n = len(all_parsed)
                ncols = min(3, n)
                nrows = int(np.ceil(n / ncols))
                fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows),
                                          squeeze=False)
                for i, p in enumerate(all_parsed):
                    r, c = i // ncols, i % ncols
                    ax = axes[r][c]
                    df = p["sim"]["data"]
                    ax.plot(df[x_col], df[y_col], color=batch_color, linewidth=0.9)
                    ax.set_title(p["name"], fontsize=9)
                    ax.set_xlabel(friendly(x_col), fontsize=8)
                    ax.set_ylabel(friendly(y_col), fontsize=8)
                    ax.tick_params(labelsize=7)
                    for spine in ax.spines.values():
                        spine.set_linewidth(0.6)
                for j in range(n, nrows * ncols):
                    axes[j // ncols][j % ncols].axis('off')

                plt.tight_layout()
                st.pyplot(fig)

                png_buf = io.BytesIO()
                fig.savefig(png_buf, format='png', dpi=300, bbox_inches='tight')
                pdf_buf = io.BytesIO()
                fig.savefig(pdf_buf, format='pdf', bbox_inches='tight')
                dcol1, dcol2 = st.columns(2)
                with dcol1:
                    st.download_button("Download PNG", png_buf.getvalue(),
                                        "batch_grid.png", "image/png")
                with dcol2:
                    st.download_button("Download PDF", pdf_buf.getvalue(),
                                        "batch_grid.pdf", "application/pdf")

# ============================================================
# SINGLE FILE MODE
# ============================================================

else:
    uploaded_file = st.file_uploader("Upload a SCAPS .iv file", type=["iv", "txt"])

    if uploaded_file is not None:
        simulations = parse_iv_file(uploaded_file.read())

        if not simulations:
            st.error("No valid simulation data found in this file.")
        else:
            st.success(f"Found {len(simulations)} simulation(s) in this file.")

            # --- Feature: overlay multiple simulations ---
            if len(simulations) > 1:
                sim_labels = [f"Simulation #{s['sim_number']}" for s in simulations]
                selected_labels = st.multiselect(
                    "Select simulation(s) to plot (choose 2+ to overlay)",
                    sim_labels, default=[sim_labels[0]])
                selected_sims = [simulations[sim_labels.index(lbl)] for lbl in selected_labels]
            else:
                selected_sims = simulations

            if not selected_sims:
                st.info("Select at least one simulation above.")
                st.stop()

            all_columns = list(selected_sims[0]["data"].columns)
            voltage_col_guess = next((c for c in all_columns if c.startswith("v(")),
                                      all_columns[0])
            x_col = st.selectbox("X-axis parameter", all_columns,
                                  index=all_columns.index(voltage_col_guess))

            jtot_col = next((c for c in all_columns if "jtot" in c), None)
            y_candidates = [c for c in all_columns if c != x_col]
            default_sel = [jtot_col] if jtot_col else y_candidates[:1]
            y_cols = st.multiselect("Y-axis parameter(s)", y_candidates,
                                     format_func=friendly, default=default_sel)

            if not y_cols:
                st.info("Select at least one Y-axis parameter.")
                st.stop()

            # Build the list of series to plot: one per (simulation, column)
            multi_sim = len(selected_sims) > 1
            series_list = []
            for sim in selected_sims:
                for ycol in y_cols:
                    label = (f"Sim#{sim['sim_number']} - {friendly(ycol)}"
                             if multi_sim else friendly(ycol))
                    series_list.append({"df": sim["data"], "ycol": ycol,
                                         "label": label, "summary": sim["summary"]})

            # --- Feature: per-curve color and line style ---
            st.write("**Per-curve styling**")
            for i, s in enumerate(series_list):
                c1, c2, c3 = st.columns([2, 1, 1])
                with c1:
                    st.write(s["label"])
                with c2:
                    s["color"] = st.color_picker(f"Color##{i}",
                                                  BRIGHT_PALETTE[i % len(BRIGHT_PALETTE)],
                                                  label_visibility="collapsed")
                with c3:
                    style_name = st.selectbox(f"Style##{i}", list(LINESTYLE_MAP.keys()),
                                               label_visibility="collapsed")
                    s["linestyle"] = LINESTYLE_MAP[style_name]

            # --- General options ---
            oc1, oc2 = st.columns(2)
            with oc1:
                log_scale = st.checkbox("Log scale on Y-axis", value=False)
                show_summary = st.checkbox("Show Voc/Jsc/FF/Efficiency box", value=True)
            with oc2:
                show_voc_line = st.checkbox("Show Voc marker (dashed line)", value=False)
                show_ticks_and_grid = st.checkbox("Show minor tick marks and gridlines",
                                                   value=False)

            y_min_val, y_max_val = None, None
            if not log_scale:
                rcol1, rcol2 = st.columns(2)
                with rcol1:
                    y_min_val = st.number_input("Y-axis minimum", value=-35.0, step=5.0)
                with rcol2:
                    y_max_val = st.number_input("Y-axis maximum", value=80.0, step=5.0)

            # --- Feature: custom axis labels ---
            st.write("**Custom axis labels**")
            lc1, lc2 = st.columns(2)
            with lc1:
                custom_xlabel = st.text_input("X-axis label", value="Voc (V)")
            with lc2:
                custom_ylabel = st.text_input("Y-axis label", value="Current Density (mA/cm\u00b2)")

            # --- Feature: adjustable font size ---
            font_size = st.slider("Overall font size", min_value=7, max_value=16, value=10)

            # --- Feature: annotation tool ---
            with st.expander("Add custom annotations"):
                ac1, ac2, ac3, ac4 = st.columns([2, 1, 1, 1])
                with ac1:
                    ann_text = st.text_input("Annotation text", key="ann_text")
                with ac2:
                    ann_x = st.number_input("X position", value=0.5, key="ann_x")
                with ac3:
                    ann_y = st.number_input("Y position", value=0.0, key="ann_y")
                with ac4:
                    st.write("")
                    if st.button("Add"):
                        if ann_text:
                            st.session_state.annotations.append(
                                {"text": ann_text, "x": ann_x, "y": ann_y})
                if st.session_state.annotations:
                    st.write(f"{len(st.session_state.annotations)} annotation(s) added.")
                    if st.button("Clear all annotations"):
                        st.session_state.annotations = []

            if st.button("Generate Graph", type="primary"):
                fig, ax = plt.subplots(figsize=(9, 6.5), dpi=150)
                plt.rcParams['font.size'] = font_size

                for s in series_list:
                    ax.plot(s["df"][x_col], s["df"][s["ycol"]], linewidth=0.9,
                            color=s["color"], linestyle=s["linestyle"],
                            label=s["label"], zorder=4)

                if show_voc_line:
                    voc_val = find_voc(series_list[0]["df"], x_col)
                    if voc_val is not None:
                        ax.axvline(x=voc_val, color='#0057ff', linestyle='--',
                                   linewidth=0.8, zorder=3, alpha=0.85)

                apply_axis_styling(ax, x_col, font_size, show_ticks_and_grid,
                                    log_scale, y_min_val, y_max_val)
                if log_scale:
                    ax.set_yscale('log')

                ax.set_xlabel(custom_xlabel, fontsize=font_size)
                ax.set_ylabel(custom_ylabel, fontsize=font_size)

                if show_voc_line:
                    voc_val = find_voc(series_list[0]["df"], x_col)
                    if voc_val is not None:
                        y_top = ax.get_ylim()[1]
                        ax.annotate(f"Voc = {voc_val:.4f} V", xy=(voc_val, y_top),
                                    xytext=(8, -8), textcoords='offset points',
                                    fontsize=font_size * 0.85, color='#0057ff',
                                    ha='left', va='top',
                                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                                              edgecolor='#0057ff', linewidth=0.5, alpha=0.95))

                # Summary box + legend, stacked top-left
                legend_anchor_y = 0.985
                first_summary = series_list[0]["summary"]
                if show_summary and first_summary:
                    lines = [f"{k} = {first_summary[k]:.4f}" for k in
                             ["Voc", "Jsc", "FF", "eta"] if k in first_summary]
                    if lines:
                        ax.text(0.015, 0.985, "\n".join(lines), transform=ax.transAxes,
                                fontsize=font_size * 0.8, va='top',
                                bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                                          edgecolor='gray', linewidth=0.6, alpha=0.95))
                        legend_anchor_y = 0.985 - (0.043 * len(lines)) - 0.035

                ax.legend(fontsize=font_size * 0.85, loc='upper left', framealpha=0.95,
                          bbox_to_anchor=(0.015, legend_anchor_y), edgecolor='gray')

                # Custom annotations
                for ann in st.session_state.annotations:
                    ax.annotate(ann["text"], xy=(ann["x"], ann["y"]), fontsize=font_size * 0.85,
                                ha='center',
                                bbox=dict(boxstyle='round,pad=0.25', facecolor='lightyellow',
                                          edgecolor='gray', linewidth=0.5, alpha=0.9))

                plt.tight_layout()
                st.pyplot(fig)

                # --- Feature: auto-generated figure caption ---
                col_names = ", ".join(friendly(s["ycol"]) for s in
                                       {id(s["ycol"]): s for s in series_list}.values())
                caption_parts = [f"Figure: {col_names} vs {friendly(x_col)}"]
                if first_summary:
                    stats = ", ".join(f"{k} = {first_summary[k]:.4f}" for k in
                                       ["Voc", "Jsc", "FF", "eta"] if k in first_summary)
                    if stats:
                        caption_parts.append(f"({stats})")
                auto_caption = " ".join(caption_parts) + "."
                st.text_area("Suggested figure caption (editable)", value=auto_caption,
                              height=70)

                png_buf = io.BytesIO()
                fig.savefig(png_buf, format='png', dpi=300, bbox_inches='tight')
                pdf_buf = io.BytesIO()
                fig.savefig(pdf_buf, format='pdf', bbox_inches='tight')

                dcol1, dcol2 = st.columns(2)
                with dcol1:
                    st.download_button("Download PNG", png_buf.getvalue(),
                                        "scaps_graph.png", "image/png")
                with dcol2:
                    st.download_button("Download PDF (publication)", pdf_buf.getvalue(),
                                        "scaps_graph.pdf", "application/pdf")

            with st.expander("View raw parsed data table"):
                st.dataframe(selected_sims[0]["data"])
