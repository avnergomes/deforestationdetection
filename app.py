import streamlit as st
import ee
import pandas as pd
import numpy as np
import plotly.express as px
import urllib.request
import tempfile
from PIL import Image as PILImage, ImageDraw
import math
import os

# ==============================================================
# Streamlit App Config
# ==============================================================
st.set_page_config(
    page_title="🌿 NDVI Time Series & GIF Generator",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🌿 NDVI Time Series & Annual GIF Generator")

# ==============================================================
# 1. Google Earth Engine Authentication (Streamlit-friendly)
# ==============================================================

def gee_authenticate_interactively():
    """
    Manual authentication flow for Earth Engine inside Streamlit.
    Opens a Go-link and provides a text box for pasting the code.
    """
    from google.auth.transport.requests import Request
    from google_auth_oauthlib.flow import InstalledAppFlow
    import ee

    st.warning("🔐 Earth Engine requires authentication. Please sign in below:")

    # Generate manual authentication URL
    try:
        auth_url = ee.oauth.get_authorization_url()
        st.markdown(
            f"👉 **[Click here to open the Google Authentication page]({auth_url})**",
            unsafe_allow_html=True
        )
        st.info("After you sign in, copy the code and paste it below 👇")
        auth_code = st.text_input("🔑 Paste the authentication code here:")

        if auth_code:
            try:
                ee.oauth.get_access_token(auth_code)
                ee.Initialize()
                st.success("✅ Google Earth Engine successfully initialized!")
                st.session_state['gee_initialized'] = True
                return True
            except Exception as e:
                st.error(f"❌ Authentication failed: {e}")
                return False
    except Exception as e:
        st.error(f"⚠️ Could not generate authentication URL: {e}")
    return False


# Try automatic initialization first
if 'gee_initialized' not in st.session_state:
    try:
        ee.Initialize()
        st.session_state['gee_initialized'] = True
        st.sidebar.success("✅ GEE already initialized.")
    except Exception:
        st.session_state['gee_initialized'] = False

# If not yet authenticated, run interactive flow
if not st.session_state['gee_initialized']:
    success = gee_authenticate_interactively()
    if not success:
        st.stop()


# ==============================================================
# 2. Sidebar Inputs
# ==============================================================
st.sidebar.header("⚙️ Configuration")
start_year = st.sidebar.number_input("Start Year", 2015, 2025, 2015)
end_year = st.sidebar.number_input("End Year", start_year, 2025, 2024)
radius = st.sidebar.slider("Radius (m)", 200, 2000, 800, 100)
dimension = st.sidebar.slider("GIF dimension (px)", 100, 500, 250, 50)

# ==============================================================
# 3. Upload CSV
# ==============================================================
st.write("""
Upload a CSV file with columns:
`name`, `latitude`, `longitude`, `description`
""")
uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)
    locations = []
    for i, row in df.iterrows():
        point = ee.Geometry.Point([row["longitude"], row["latitude"]])
        name = row.get("name", f"Location_{i+1}")
        locations.append({"name": name, "geometry": point})

    # ==============================================================
    # 4. Helper Functions
    # ==============================================================
    def mask_clouds(image):
        qa = image.select("QA_PIXEL")
        mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
        return image.updateMask(mask)

    def add_ndvi(image):
        scaled = image.select(["SR_B4", "SR_B5"]).multiply(0.0000275).add(-0.2)
        ndvi = scaled.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")
        return image.addBands(ndvi)

    def get_ndvi_time_series(point, region_name):
        col = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
            .filterBounds(point)
            .map(mask_clouds)
            .map(add_ndvi)
            .select("NDVI")
        )
        def extract(image):
            mean = image.reduceRegion(ee.Reducer.mean(), point, 30)
            return ee.Feature(None, {
                "NDVI": mean.get("NDVI"),
                "date": ee.Date(image.get("system:time_start")).format("YYYY-MM-dd")
            })
        fc = col.map(extract).filter(ee.Filter.notNull(["NDVI"]))
        feats = fc.getInfo().get("features", [])
        df = pd.DataFrame([
            {"date": f["properties"]["date"], "NDVI": f["properties"]["NDVI"]}
            for f in feats if f["properties"].get("NDVI") is not None
        ])
        df["date"] = pd.to_datetime(df["date"])
        df["region"] = region_name
        return df

    def generate_and_save_gif(location, name):
        all_years = list(range(start_year, end_year + 1))
        yearly_images, frame_years = [], []
        for year in all_years:
            start = f"{year}-01-01"
            end = f"{year}-12-31"
            col = (
                ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
                .filterBounds(location)
                .filterDate(start, end)
                .map(mask_clouds)
                .map(add_ndvi)
                .select("NDVI")
            )
            image = col.median().set("system:time_start", ee.Date(start).millis())
            yearly_images.append(image)
            frame_years.append(year)

        annual_collection = ee.ImageCollection(yearly_images)
        vis = {"min": 0, "max": 1, "palette": ["white", "lightgreen", "green", "darkgreen"]}
        region = location.buffer(radius)
        gif_params = {
            "dimensions": dimension,
            "framesPerSecond": 1,
            "region": region,
            "format": "gif",
            **vis,
        }
        gif_url = annual_collection.getVideoThumbURL(gif_params)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".gif")
        urllib.request.urlretrieve(gif_url, temp_file.name)
        return temp_file.name, frame_years

    def add_scale_bar(frame, gif_dim=250, radius_m=800):
        draw = ImageDraw.Draw(frame)
        width, height = frame.width, frame.height
        meters_per_pixel = (radius_m * 2) / gif_dim
        raw_len = (radius_m * 2) / 5
        scale_m = 10 ** math.floor(math.log10(raw_len))
        if raw_len / scale_m > 5:
            scale_m *= 5
        elif raw_len / scale_m > 2:
            scale_m *= 2
        bar_px = int(scale_m / meters_per_pixel)
        margin = 10
        y0 = height - 20
        x0 = width - bar_px - margin
        x1 = width - margin
        draw.rectangle([x0, y0, x1, y0 + 8], fill="white", outline="black", width=1)
        draw.text((x0, y0 - 15), f"{int(scale_m)} m", fill="white")
        return frame

    def annotate_gif_with_years(gif_path, frame_years):
        img = PILImage.open(gif_path)
        frames = []
        n = img.n_frames
        frame_years = frame_years[:n]
        for i in range(n):
            img.seek(i)
            frame = img.convert("RGB")
            draw = ImageDraw.Draw(frame)
            draw.rectangle([0, frame.height - 25, 100, frame.height], fill=(0, 0, 0))
            draw.text((5, frame.height - 20), f"Year: {frame_years[i]}", fill="white")
            frame = add_scale_bar(frame, dimension, radius)
            frames.append(frame)
        output_file = gif_path.replace(".gif", "_annotated.gif")
        frames[0].save(output_file, save_all=True, append_images=frames[1:], loop=0, duration=1000)
        return output_file

    # ==============================================================
    # 5. Run button
    # ==============================================================
    if st.button("🚀 Run NDVI Analysis"):
        results = []
        with st.spinner("Processing NDVI data... this may take several minutes ⏳"):
            for loc in locations:
                st.write(f"📍 Processing {loc['name']}...")
                df_loc = get_ndvi_time_series(loc["geometry"], loc["name"])
                results.append(df_loc)
            ndvi_all = pd.concat(results).sort_values("date").reset_index(drop=True)
            csv_path = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
            ndvi_all.to_csv(csv_path, index=False)
            st.success("✅ NDVI data collected successfully!")

            # --- Plot NDVI time series
            fig = px.line(ndvi_all, x="date", y="NDVI", color="region",
                          title="NDVI Time Series per Location")
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("⬇️ Download NDVI CSV", open(csv_path, "rb"), "ndvi_timeseries.csv")

            # --- Generate GIFs
            st.write("🌀 Generating annual NDVI GIFs...")
            for loc in locations:
                gif_file, frame_years = generate_and_save_gif(loc["geometry"], loc["name"])
                annotated = annotate_gif_with_years(gif_file, frame_years)
                st.image(annotated, caption=f"NDVI Annual GIF – {loc['name']}", use_column_width=True)
