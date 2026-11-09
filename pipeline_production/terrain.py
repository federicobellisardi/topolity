"""Shared terrain and geometry helpers for the global and densification pipelines.

Elevation validity rule (used everywhere elevations enter an energy term)
------------------------------------------------------------------------
* Values >= Z_MIN_VALID_M are valid terrain, including genuine below-sea-level
  areas (e.g. Amsterdam polders, about -9 m on the street network).
* Values < Z_MIN_VALID_M or non-finite values are DEM voids/artefacts (e.g. the
  -220 m spikes in the Brussels/Amsterdam tile and -86 m near Buenos Aires).
  Along an edge profile they are replaced by linear interpolation between the
  nearest valid samples; a profile with no valid sample is flat (zero gain).
* The GeoTIFF nodata tag is ignored: our DEMs store sea level as 0 and tag 0
  as nodata, so honouring the tag would turn every coastal 0 m pixel into a void.

The same cleaned profile feeds both the uphill gain and the 3-D length.
"""

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer
from scipy.ndimage import map_coordinates


Z_MIN_VALID_M = -30.0


def clean_elevation_profile(z):
    """Replace invalid samples by linear interpolation along the profile."""
    z = np.asarray(z, dtype=float).copy()
    valid = np.isfinite(z) & (z >= Z_MIN_VALID_M)
    if valid.all():
        return z
    if not valid.any():
        return np.zeros_like(z)
    idx = np.arange(len(z))
    z[~valid] = np.interp(idx[~valid], idx[valid], z[valid])
    return z


class MetricFrame:
    """Local azimuthal-equidistant frame (metres) centred on (lon0, lat0).

    Rigid transformations of a lon/lat network are applied in this frame and
    mapped back, so internal distances are preserved. At the scale of a FUA
    (radius <~ 60 km) the AEQD scale error is below 1e-4.
    """

    def __init__(self, lon0, lat0):
        self.lon0, self.lat0 = float(lon0), float(lat0)
        proj = (f"+proj=aeqd +lat_0={self.lat0} +lon_0={self.lon0} "
                "+datum=WGS84 +units=m +no_defs")
        self._fwd = Transformer.from_crs("EPSG:4326", proj, always_xy=True)
        self._inv = Transformer.from_crs(proj, "EPSG:4326", always_xy=True)

    @classmethod
    def from_graph(cls, G):
        """Frame centred on the bounding-box centre of the graph nodes."""
        xs = np.fromiter((d["x"] for _, d in G.nodes(data=True)), float)
        ys = np.fromiter((d["y"] for _, d in G.nodes(data=True)), float)
        return cls(0.5 * (xs.min() + xs.max()), 0.5 * (ys.min() + ys.max()))

    def to_metric(self, lon, lat):
        x, y = self._fwd.transform(np.asarray(lon, float), np.asarray(lat, float))
        return np.asarray(x, float), np.asarray(y, float)

    def to_lonlat(self, x, y):
        lon, lat = self._inv.transform(np.asarray(x, float), np.asarray(y, float))
        return np.asarray(lon, float), np.asarray(lat, float)

    def apply(self, lon, lat, fn):
        """Map lon/lat -> metres -> fn(x, y) -> lon/lat."""
        x, y = self.to_metric(lon, lat)
        return self.to_lonlat(*fn(x, y))


def translation_fn(dx_m, dy_m):
    return lambda x, y: (x + dx_m, y + dy_m)


def rotation_fn(angle_deg):
    """Counter-clockwise rotation about the frame origin."""
    th = np.radians(angle_deg)
    c, s = np.cos(th), np.sin(th)
    return lambda x, y: (c * x - s * y, s * x + c * y)


def scaling_fn(factor, axis="both"):
    """Scaling about the frame origin; axis 'x' = E-W, 'y' = N-S."""
    fx = factor if axis in ("x", "both") else 1.0
    fy = factor if axis in ("y", "both") else 1.0
    return lambda x, y: (fx * x, fy * y)


def resample_polyline(x, y, ds):
    """Points every ~ds metres along a metric polyline (endpoints included)."""
    seg = np.hypot(np.diff(x), np.diff(y))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    length = float(cum[-1])
    n = max(int(np.ceil(length / ds)) + 1, 2)
    s = np.linspace(0.0, length, n)
    return np.interp(s, cum, x), np.interp(s, cum, y)


class GeoDEM:
    """In-memory window of a lon/lat DEM, sampled bilinearly.

    Invalid pixels (< Z_MIN_VALID_M) are set to NaN before interpolation, so
    they never leak into neighbouring samples; the NaNs are then filled along
    each profile by `clean_elevation_profile`.
    """

    def __init__(self, dem_file, bounds, margin_deg=0.01):
        minx, miny, maxx, maxy = bounds
        with rasterio.open(dem_file) as src:
            if src.crs is not None and src.crs.to_epsg() != 4326:
                raise ValueError(f"GeoDEM expects an EPSG:4326 DEM, got {src.crs}")
            win = from_bounds(minx - margin_deg, miny - margin_deg,
                              maxx + margin_deg, maxy + margin_deg, src.transform)
            win = win.round_offsets().round_lengths()
            arr = src.read(1, window=win, boundless=True, fill_value=np.nan,
                           out_dtype="float32", masked=False)
            self.transform = src.window_transform(win)
        arr[~np.isfinite(arr) | (arr < Z_MIN_VALID_M)] = np.nan
        self.arr = arr
        self.inv = ~self.transform

    def sample(self, lon, lat):
        inv = self.inv
        lon = np.asarray(lon, float)
        lat = np.asarray(lat, float)
        # affine gives pixel-corner coordinates; map_coordinates uses pixel centres
        col = inv.a * lon + inv.b * lat + inv.c - 0.5
        row = inv.d * lon + inv.e * lat + inv.f - 0.5
        return map_coordinates(self.arr, [row, col], order=1, mode="constant",
                               cval=np.nan, prefilter=False)
