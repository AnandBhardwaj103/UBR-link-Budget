
# --- Imports ---
import streamlit as st
import math
import pandas as pd
import numpy as np
import folium
from folium import Map, Marker, TileLayer, Polygon
import branca.colormap as cm

# --- Page config must be first ---
st.set_page_config(page_title="Airtel Link Budget and Coverage tool", layout="wide")

# --- Initialise session state variables ---
if 'clicked_points' not in st.session_state:
    st.session_state.clicked_points = []
if 'cpe_noise_list' not in st.session_state:
    st.session_state.cpe_noise_list = []

# --- Sidebar Inputs ---
st.sidebar.image("airtel_logo.png", use_container_width=True)
st.image("airtel_logo.png", width=100)
st.title("Airtel Link Budget and Coverage tool")

st.sidebar.header("Access Point Settings")
freq_ghz = 5.0
ap_lat = st.sidebar.number_input("AP latitude", value=28.6139, format="%.6f")
ap_lon = st.sidebar.number_input("AP longitude", value=77.2090, format="%.6f")
ap_tx_dbm = st.sidebar.number_input("AP Tx power (dBm)", value=23.0)
ap_gmax_dbi = st.sidebar.number_input("AP antenna Gmax (dBi)", value=5.0)
ap_az = st.sidebar.slider("AP azimuth (deg)", 0, 359, 0)
ap_bw = st.sidebar.number_input("AP 3dB beamwidth (deg)", value=90.0, min_value=5.0, max_value=180.0)

# --- AP Spectrum Noise/Interference (dBm) ---
ap_noise_dbm = st.sidebar.slider(
    "AP Spectrum Noise / Interference (dBm)",
    min_value=-95, max_value=-25, value=-95, step=1
)
st.sidebar.markdown("---")

st.sidebar.header("CPE Parameters")
ue_gain_dbi = st.sidebar.number_input("CPE Antenna Gmax (dBi)", value=2.0)
ue_bw = st.sidebar.number_input("CPE 3dB beamwidth (deg)", value=90.0, min_value=5.0, max_value=180.0)
ue_az_default = 180
cpe_tx_dbm = st.sidebar.number_input("CPE Tx power (dBm)", value=20.0)
st.sidebar.markdown("**Note:** CPE EIRP = Tx power + Antenna gain")
st.sidebar.markdown("---")

st.sidebar.header("Channel & Noise")
bw_mhz = st.sidebar.selectbox("Channel bandwidth (MHz)", [40, 80, 160], index=1)
noise_figure_db = st.sidebar.number_input("Noise figure (dB)", value=6.0)
impl_loss_db = 2.0
num_streams = st.sidebar.selectbox("Number of spatial streams", [1, 2, 4, 8], index=1)
num_ues = st.sidebar.slider("Number of CPEs to place", min_value=1, max_value=16, value=5)
environment = st.sidebar.selectbox("Environment", ["Indoor", "Outdoor"], index=0)
condition = st.sidebar.selectbox("Condition", ["LOS", "NLOS"], index=0)
fading_loss_db = st.sidebar.number_input("Fading Loss (dB)", value=0.0, min_value=0.0, max_value=30.0)
obss_penalty_db = st.sidebar.slider("OBSS / Interference Penalty (dB)", min_value=0, max_value=20, value=0)
efficiency = st.sidebar.slider("MAC Efficiency (%)", min_value=50, max_value=90, value=70) / 100.0
st.sidebar.markdown("---")

st.sidebar.header("TDD Ratio")
tdd_ratio_dl = st.sidebar.slider("DL Ratio (%)", min_value=10, max_value=90, value=70)
tdd_ratio_ul = 100 - tdd_ratio_dl
st.sidebar.write(f"UL Ratio: {tdd_ratio_ul}%")

if st.sidebar.button("Reset CPEs"):
    st.session_state.clicked_points = []
    st.session_state.cpe_noise_list = []
    st.toast("CPE list cleared", icon="✅")

# --- Helper Functions ---
def noise_floor_dbm(bandwidth_hz, nf_db=6.0, impl_loss_db=1.0):
    thermal_dbm = -174.0 + 10 * math.log10(bandwidth_hz)
    return thermal_dbm + nf_db + impl_loss_db

def snr_db(received_dbm, noise_dbm):
    return received_dbm - noise_dbm

def beam_gain_3gpp(az_offset_deg, bw_3db_deg, Gmax_dbi, A_m=25.0):
    # 3GPP-like azimuth-only pattern
    val = 12.0 * (az_offset_deg / max(bw_3db_deg, 0.1)) ** 2
    return Gmax_dbi - min(val, A_m)

def bearing_deg(lon1, lat1, lon2, lat2):
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δλ = math.radians(lon2 - lon1)
    y = math.sin(Δλ) * math.cos(φ2)
    x = math.cos(φ1)*math.sin(φ2) - math.sin(φ1)*math.cos(φ2)*math.cos(Δλ)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0

def clamp_rssi(dbm, min_dbm=-92.0):
    # You can loosen to -110 dBm if you want to see deeper edge values
    return max(dbm, min_dbm)

def get_pathloss_exponent(env="Indoor", cond="LOS"):
    if env == "Indoor":
        return 2.0 if cond == "LOS" else 2.8
    else:
        return 2.0 if cond == "LOS" else 3.0

def path_loss_wifi(distance_m, freq_ghz, env="Indoor", cond="LOS"):
    # distance floor to avoid near-field oddities
    d_m = max(distance_m, 1.0)
    n = get_pathloss_exponent(env, cond)
    return 32.4 + 10 * n * math.log10(d_m) + 20 * math.log10(freq_ghz)

MCS_TABLES = {
    40: [8.6, 17.2, 25.8, 34.4, 51.6, 68.8, 77.4, 103.2, 114.7, 137.6, 154.9, 172.1],
    80: [17.2, 34.4, 51.6, 68.8, 103.2, 137.6, 154.9, 206.5, 229.4, 275.3, 309.8, 344.4],
    160:[34.4, 68.8, 103.2, 137.6, 206.5, 275.3, 309.8, 413.0, 458.8, 550.6, 619.6, 688.8]
}
def snr_to_mcs(snr_db_val):
    thresholds = [0, 5, 8, 11, 15, 18, 20, 23, 25, 28, 30, 32]
    for i, t in enumerate(thresholds[::-1]):
        if snr_db_val >= t:
            return len(thresholds) - i - 1
    return 0

# --- Geometry helpers ---
meters_per_deg_lat = 111320.0
meters_per_deg_lon = 40075000.0 * math.cos(math.radians(ap_lat)) / 360.0
bw_hz = bw_mhz * 1e6

# --- Coverage fan with hard RSL limit (-98 dBm) ---
RSL_THRESHOLD_DBM = -98.0
pl_allow_db = ap_tx_dbm + ap_gmax_dbi + ue_gain_dbi - RSL_THRESHOLD_DBM - impl_loss_db - fading_loss_db
n_exp = get_pathloss_exponent(environment, condition)
distance_m_max = 10 ** ((pl_allow_db - 32.4 - 20 * math.log10(freq_ghz)) / (10 * n_exp))

# --- Add Next CPE seeded at sector edge ---
def _seed_cpe_on_sector_edge(idx_from_zero: int):
    half_bw = ap_bw / 2.0
    radius = 0.9 * distance_m_max
    offset_deg = (-half_bw + (idx_from_zero % max(1, int(ap_bw))))
    bearing_deg_val = ap_az + offset_deg
    bearing = math.radians(bearing_deg_val)
    dlat = (radius * math.cos(bearing)) / meters_per_deg_lat
    dlon = (radius * math.sin(bearing)) / meters_per_deg_lon
    return {'lat': ap_lat + dlat, 'lng': ap_lon + dlon, 'tx_power': cpe_tx_dbm, 'ue_az': ue_az_default}

remaining = num_ues - len(st.session_state.clicked_points)
colA, colB = st.columns([1, 1], gap="small")
with colA:
    if st.button("➕ Add Next CPE", use_container_width=True, disabled=(remaining <= 0)):
        new_idx = len(st.session_state.clicked_points)
        seed = _seed_cpe_on_sector_edge(new_idx)
        st.session_state.clicked_points.append(seed)
        remaining = max(num_ues - len(st.session_state.clicked_points), 0)
        st.toast(
            f"CPE {new_idx+1} added. " +
            (f"Click on the map to reposition, or edit in the table. "
             f"Add Next CPE to continue ({remaining} remaining)." if remaining > 0 else "All CPEs placed."),
            icon="📍"
        )
with colB:
    if remaining > 0:
        st.info(f"Remaining to place: **{remaining}** / {num_ues}")
    else:
        st.success("All requested CPEs placed. You can edit details below.")

# --- CPE Table with Spectrum Noise Input ---
st.subheader("Edit CPE Positions, Tx Power, Azimuth & Spectrum Noise")
# Keep noise list length in sync
if len(st.session_state.cpe_noise_list) < len(st.session_state.clicked_points):
    st.session_state.cpe_noise_list += [-95] * (len(st.session_state.clicked_points) - len(st.session_state.cpe_noise_list))

if st.session_state.clicked_points:
    editable_df = pd.DataFrame([
        {
            'CPE': i + 1,
            'Latitude': p.get('lat', ap_lat),
            'Longitude': p.get('lng', ap_lon),
            'Tx Power (dBm)': float(p.get('tx_power', cpe_tx_dbm)),
            'CPE Azimuth (deg)': float(p.get('ue_az', ue_az_default)),
            'CPE Spectrum Noise (dBm)': st.session_state.cpe_noise_list[i] if i < len(st.session_state.cpe_noise_list) else -95
        }
        for i, p in enumerate(st.session_state.clicked_points)
    ])

    def _sanitize_row(row):
        row['Tx Power (dBm)'] = float(np.clip(row['Tx Power (dBm)'], 0.0, 30.0))
        row['CPE Azimuth (deg)'] = float(row['CPE Azimuth (deg)']) % 360.0
        row['CPE Spectrum Noise (dBm)'] = float(np.clip(row['CPE Spectrum Noise (dBm)'], -95, -25))
        return row

    edited_df = st.data_editor(editable_df, num_rows="dynamic", key="cpe_editor")
    edited_df = edited_df.apply(_sanitize_row, axis=1)

    # Update session state with edited values
    st.session_state.clicked_points = [
        {
            'lat': float(row['Latitude']),
            'lng': float(row['Longitude']),
            'tx_power': float(row['Tx Power (dBm)']),
            'ue_az': float(row['CPE Azimuth (deg)'])
        }
        for _, row in edited_df.iterrows()
    ]
    st.session_state.cpe_noise_list = list(edited_df['CPE Spectrum Noise (dBm)'])
else:
    st.info("Use **Add Next CPE** or click on the map to start adding CPEs.")

# --- Per‑CPE PHY and throughput computation ---
ue_results = []
phy_rates_dl = []
phy_rates_ul = []
if st.session_state.clicked_points:
    n_cpe = len(st.session_state.clicked_points)

    # --- Per-UE link metrics first (RSSI/SNR/MCS/PHY) ---
    for idx, point in enumerate(st.session_state.clicked_points):
        ue_lat = point['lat']; ue_lon = point['lng']

        dy = (ue_lat - ap_lat) * meters_per_deg_lat
        dx = (ue_lon - ap_lon) * meters_per_deg_lon
        distance_m = math.hypot(dx, dy)

        # AP directional gain toward UE
        az_ap_to_ue = bearing_deg(ap_lon, ap_lat, ue_lon, ue_lat)
        az_offset_ap = min(abs((az_ap_to_ue - ap_az + 180) % 360 - 180), 180)
        G_ap_dir = beam_gain_3gpp(az_offset_ap, ap_bw, ap_gmax_dbi)

        # UE directional gain toward AP
        az_ue_to_ap = bearing_deg(ue_lon, ue_lat, ap_lon, ap_lat)
        az_offset_ue = min(abs((az_ue_to_ap - point.get('ue_az', ue_az_default) + 180) % 360 - 180), 180)
        G_ue_dir = beam_gain_3gpp(az_offset_ue, ue_bw, ue_gain_dbi)

        # Path loss
        pl_db = path_loss_wifi(distance_m, freq_ghz, environment, condition) + fading_loss_db

        # Per-CPE noise input (dBm)
        cpe_noise_dbm = st.session_state.cpe_noise_list[idx] if idx < len(st.session_state.cpe_noise_list) else -95

        # --- Downlink (AP → CPE): RSSI/SNR/MCS/PHY ---
        rx_dl_dbm = clamp_rssi(ap_tx_dbm + G_ap_dir + G_ue_dir - pl_db - impl_loss_db)
        noise_dl_dbm = max(ap_noise_dbm, cpe_noise_dbm)  # conservative choice
        snr_dl = rx_dl_dbm - noise_dl_dbm
        mcs_dl = snr_to_mcs(snr_dl)
        phy_rate_dl = MCS_TABLES[bw_mhz][mcs_dl] * num_streams
        phy_rates_dl.append(phy_rate_dl)

        # --- Uplink (CPE → AP): RSSI/SNR/MCS/PHY ---
        cpe_eirp_dir_dbm = float(point.get('tx_power', cpe_tx_dbm)) + G_ue_dir
        rx_ul_dbm = clamp_rssi(cpe_eirp_dir_dbm + G_ap_dir - pl_db - impl_loss_db)
        noise_ul_dbm = max(ap_noise_dbm, cpe_noise_dbm)  # conservative choice
        snr_ul = rx_ul_dbm - noise_ul_dbm
        mcs_ul = snr_to_mcs(snr_ul)
        phy_rate_ul = MCS_TABLES[bw_mhz][mcs_ul] * num_streams
        phy_rates_ul.append(phy_rate_ul)

        ue_results.append({
            'CPE': idx + 1,
            'lat': ue_lat,
            'lon': ue_lon,
            'distance_m': distance_m,
            'rx_dl_dbm': rx_dl_dbm,
            'snr_dl_db': snr_dl,
            'MCS_DL': mcs_dl,
            'PHY Rate DL (Mbps)': phy_rate_dl,
            'rx_ul_dbm': rx_ul_dbm,
            'snr_ul_db': snr_ul,
            'MCS_UL': mcs_ul,
            'PHY Rate UL (Mbps)': phy_rate_ul,
            'Tx Power (dBm)': float(point.get('tx_power', cpe_tx_dbm)),
            'CPE Azimuth (deg)': float(point.get('ue_az', ue_az_default)),
            'CPE Spectrum Noise (dBm)': cpe_noise_dbm
        })

    # --- AP aggregated capacity (DL+UL) based on spatial streams ---
    # For 2x2 -> 800 Mbps, for 4x4 -> 1600 Mbps (treated as MAC aggregate)
    # For other stream counts (1 or 8), fallback to table-derived estimate.
    ap_max_phy_rate_table = MCS_TABLES[bw_mhz][-1] * num_streams
    ap_fallback_mac_capacity = ap_max_phy_rate_table * efficiency  # fallback

    if num_streams == 2:
        ap_total_mac_capacity = 800.0
    elif num_streams == 4:
        ap_total_mac_capacity = 1600.0
    else:
        ap_total_mac_capacity = ap_fallback_mac_capacity

    # Split by TDD ratio
    ap_dl_capacity = ap_total_mac_capacity * (tdd_ratio_dl / 100.0)
    ap_ul_capacity = ap_total_mac_capacity * (tdd_ratio_ul / 100.0)

    # --- Target-based throughput allocator with per-UE MAC caps ---
    TARGET_DL = 40.0  # Mbps per CPE
    TARGET_UL = 10.0  # Mbps per CPE

    # Per-UE MAC caps to avoid overestimating above PHY
    cap_dl = np.array([0.7 * r for r in phy_rates_dl], dtype=float)  # 70% of PHY
    cap_ul = np.array([0.7 * r for r in phy_rates_ul], dtype=float)

    # Demand limited by per-UE caps
    demand_dl = np.minimum(TARGET_DL, cap_dl)
    demand_ul = np.minimum(TARGET_UL, cap_ul)

    # Allocate DL against AP DL capacity
    sum_demand_dl = float(np.sum(demand_dl))
    if sum_demand_dl <= ap_dl_capacity:
        alloc_dl = demand_dl
    else:
        # Proportional downscale against target demand, keep within per-UE caps
        scale = ap_dl_capacity / sum_demand_dl if sum_demand_dl > 0 else 0.0
        alloc_dl = np.minimum(cap_dl, demand_dl * scale)

    # Allocate UL against AP UL capacity
    sum_demand_ul = float(np.sum(demand_ul))
    if sum_demand_ul <= ap_ul_capacity:
        alloc_ul = demand_ul
    else:
        scale = ap_ul_capacity / sum_demand_ul if sum_demand_ul > 0 else 0.0
        alloc_ul = np.minimum(cap_ul, demand_ul * scale)

    # Attach allocated throughputs to UE results
    for i, ue in enumerate(ue_results):
        ue['Target DL (Mbps)'] = TARGET_DL
        ue['Target UL (Mbps)'] = TARGET_UL
        ue['DL Throughput (Mbps)'] = round(float(alloc_dl[i]), 2)
        ue['UL Throughput (Mbps)'] = round(float(alloc_ul[i]), 2)
        ue['DL meets target'] = bool(alloc_dl[i] >= TARGET_DL - 1e-6)
        ue['UL meets target'] = bool(alloc_ul[i] >= TARGET_UL - 1e-6)

# --- Map Rendering with coverage fan and rich tooltips ---
# Build map after computing ue_results so tooltips have metrics
m = Map(location=[ap_lat, ap_lon], zoom_start=17)
TileLayer('cartodbpositron').add_to(m)
Marker([ap_lat, ap_lon], tooltip='Access Point', icon=folium.Icon(color='blue', icon='wifi')).add_to(m)

# Coverage fan (RSL = -98 dBm limit)
steps = 10
half_bw = ap_bw / 2.0
for i in range(steps):
    radius = distance_m_max * (i + 1) / steps
    color = cm.LinearColormap(['green', 'yellow', 'orange', 'red'], vmin=0, vmax=steps)(i)
    points = []
    num_points = max(8, int(ap_bw))
    for a in np.linspace(-half_bw, half_bw, num_points):
        bearing = math.radians(ap_az + a)
        dlat = (radius * math.cos(bearing)) / meters_per_deg_lat
        dlon = (radius * math.sin(bearing)) / meters_per_deg_lon
        points.append((ap_lat + dlat, ap_lon + dlon))
    Polygon(
        [(ap_lat, ap_lon)] + points + [(ap_lat, ap_lon)],
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.15
    ).add_to(m)

# CPE markers with tooltips (show metrics on hover)
if st.session_state.clicked_points:
    for idx, p in enumerate(st.session_state.clicked_points):
        if idx < len(ue_results):
            ue = ue_results[idx]
            dl_ok = ue.get('DL meets target', False)
            ul_ok = ue.get('UL meets target', False)
            marker_color = 'green' if (dl_ok and ul_ok) else 'red'
            tooltip_html = f"""
            <div style="font-size:12px;">
              <b>CPE {ue['CPE']}</b><br/>
              Distance: {ue['distance_m']:.1f} m<br/>
              DL RSSI: {ue['rx_dl_dbm']:.1f} dBm &nbsp;|&nbsp; UL RSSI: {ue['rx_ul_dbm']:.1f} dBm<br/>
              DL SNR: {ue['snr_dl_db']:.1f} dB &nbsp;|&nbsp; UL SNR: {ue['snr_ul_db']:.1f} dB<br/>
              DL MCS: {ue['MCS_DL']} &nbsp;|&nbsp; UL MCS: {ue['MCS_UL']}<br/>
              DL PHY: {ue['PHY Rate DL (Mbps)']:.1f} Mbps &nbsp;|&nbsp; UL PHY: {ue['PHY Rate UL (Mbps)']:.1f} Mbps<br/>
              Alloc DL: {ue['DL Throughput (Mbps)']:.2f} Mbps (target {ue['Target DL (Mbps)']} Mbps)<br/>
              Alloc UL: {ue['UL Throughput (Mbps)']:.2f} Mbps (target {ue['Target UL (Mbps)']} Mbps)<br/>
            </div>
            """
        else:
            marker_color = 'orange'
            tooltip_html = f"<div style='font-size:12px;'><b>CPE {idx+1}</b><br/>No metrics yet.</div>"

        Marker(
            [p['lat'], p['lng']],
            tooltip=folium.Tooltip(tooltip_html, sticky=True),
            icon=folium.Icon(color=marker_color, icon='signal')
        ).add_to(m)

# --- Interactive map handling ---
try:
    from streamlit_folium import st_folium
    SF_AVAILABLE = True
except Exception:
    SF_AVAILABLE = False

if SF_AVAILABLE:
    result = st_folium(m, width=800, height=600, key="map")
    remaining = num_ues - len(st.session_state.clicked_points)
    if result and result.get('last_clicked') and remaining > 0:
        lat = float(result['last_clicked']['lat'])
        lng = float(result['last_clicked']['lng'])
        st.session_state.clicked_points.append({'lat': lat, 'lng': lng, 'tx_power': cpe_tx_dbm, 'ue_az': ue_az_default})
        placed = len(st.session_state.clicked_points)
        remaining = max(num_ues - placed, 0)
        st.toast(
            f"CPE {placed} added. " +
            (f"Click again to add CPE {placed+1} of {num_ues}." if remaining > 0 else "All CPEs placed."),
            icon="📍"
        )
else:
    st.components.v1.html(m._repr_html_(), height=600)
    st.info("Map is non‑interactive on this machine. Use the **Add Next CPE** button or the table to add/edit CPEs.")

# --- Detailed Metrics, CSV, Charts ---
if st.session_state.clicked_points and len(ue_results) == len(st.session_state.clicked_points):
    st.subheader("Detailed Metrics")
    df = pd.DataFrame(ue_results)
    st.dataframe(df[[
        'CPE', 'lat', 'lon', 'distance_m',
        'rx_dl_dbm', 'snr_dl_db', 'MCS_DL', 'PHY Rate DL (Mbps)', 'DL Throughput (Mbps)', 'DL meets target',
        'rx_ul_dbm', 'snr_ul_db', 'MCS_UL', 'PHY Rate UL (Mbps)', 'UL Throughput (Mbps)', 'UL meets target',
        'Tx Power (dBm)', 'CPE Azimuth (deg)', 'CPE Spectrum Noise (dBm)',
        'Target DL (Mbps)', 'Target UL (Mbps)'
    ]])

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Metrics as CSV",
        data=csv,
        file_name="cpe_metrics_wifi6.csv",
        mime="text/csv"
    )

    st.subheader("DL Throughput per CPE")
    st.bar_chart(df.set_index('CPE')['DL Throughput (Mbps)'])
    st.subheader("UL Throughput per CPE")
    st.bar_chart(df.set_index('CPE')['UL Throughput (Mbps)'])

    st.caption("Coverage mask limited at RSL = −98 dBm (envelope using AP/CPE Gmax).")
else:
    if not st.session_state.clicked_points:
        st.warning("No CPEs yet. Use **Add Next CPE** or click on the map (if interactive). A toast will guide you to add the next one.")
