#!/usr/bin/env python3
"""
TESS Input Catalog (TIC) Data Query, Cleaning, and Filtering Pipeline.

This script queries TESS targets for a specific sector, retrieves their stellar properties
from the MAST TIC catalog, joins Gaia DR3 RUWE astrometric quality metrics, cleans the dataset,
performs physical imputations for missing radius and mass, and filters targets using
a set of 11 astrophysical pre-sieve filters.

Author: Senior Exoplanet Data Scientist & Python Developer
"""

import argparse
import sys
import logging
import time
from pathlib import Path
import numpy as np
import pandas as pd

# Keep remote-only dependencies lazy.  Offline operations (listing filters and
# processing a downloaded TIC CSV) should not require the large astroquery stack.

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Column order used by the 125-column bulk TIC CSV downloads.  MAST bulk files
# commonly omit the header, as does the hackathon input supplied with this repo.
TIC_COLUMNS = """ID version HIP TYC UCAC TWOMASS SDSS ALLWISE GAIA APASS KIC objType typeSrc ra dec POSflag pmRA e_pmRA pmDEC e_pmDEC PMflag plx e_plx PARflag gallong gallat eclong eclat Bmag e_Bmag Vmag e_Vmag umag e_umag gmag e_gmag rmag e_rmag imag e_imag zmag e_zmag Jmag e_Jmag Hmag e_Hmag Kmag e_Kmag TWOMflag prox w1mag e_w1mag w2mag e_w2mag w3mag e_w3mag w4mag e_w4mag GAIAmag e_GAIAmag Tmag e_Tmag TESSflag SPFlag Teff e_Teff logg e_logg MH e_MH rad e_rad mass e_mass rho e_rho lumclass lum e_lum d e_d ebv e_ebv numcont contratio disposition duplicate_id priority eneg_EBV epos_EBV EBVflag eneg_Mass epos_Mass eneg_Rad epos_Rad eneg_rho epos_rho eneg_logg epos_logg eneg_lum epos_lum eneg_dist epos_dist distflag eneg_Teff epos_Teff TeffFlag gaiabp e_gaiabp gaiarp e_gaiarp gaiaqflag starchareFlag VmagFlag BmagFlag splists e_RA e_Dec RA_orig Dec_orig e_RA_orig e_Dec_orig raddflag wdflag objID""".split()

COMPACT_COLUMNS = [
    'ID', 'GAIA', 'objType', 'ra', 'dec', 'plx', 'Tmag', 'Teff', 'logg',
    'MH', 'feh', 'rad', 'mass', 'lum', 'd', 'contratio', 'disposition',
    'priority', 'ruwe'
]


def _import_mast():
    try:
        from astroquery.mast import Catalogs, Observations
    except ImportError as exc:
        raise RuntimeError(
            "Remote queries require astroquery. Install it with "
            "'python -m pip install astroquery', or use --input-csv."
        ) from exc
    return Catalogs, Observations


def _import_gaia():
    try:
        from astroquery.gaia import Gaia
    except ImportError as exc:
        raise RuntimeError(
            "Gaia enrichment requires astroquery. Install it with "
            "'python -m pip install astroquery'."
        ) from exc
    return Gaia


def _tic_csv_kwargs(path: str, compact: bool = False) -> tuple[dict, str]:
    """Build safe pandas arguments for a headered or standard headerless TIC CSV."""
    with open(path, 'r', encoding='utf-8-sig') as handle:
        first_line = handle.readline().strip()
        first_value = first_line.split(',', 1)[0].strip().upper()
    has_header = first_value == 'ID'
    kwargs = {'low_memory': False}
    if not has_header:
        kwargs.update(header=None, names=TIC_COLUMNS)
    if compact:
        available = first_line.split(',') if has_header else TIC_COLUMNS
        # Optional fields such as RUWE and feh are retained when present without
        # making them mandatory for older/headerless TIC bulk exports.
        kwargs['usecols'] = [column for column in COMPACT_COLUMNS if column in available]
    return kwargs, 'headered' if has_header else '125-column headerless'


def _normalize_catalog_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Expose canonical filter columns while preserving catalog-specific aliases."""
    if 'MH' not in df.columns and 'feh' in df.columns:
        df['MH'] = df['feh']
    return df


def read_tic_csv(path: str, compact: bool = False) -> pd.DataFrame:
    """Read a headered or headerless TIC bulk CSV, optionally with fewer columns."""
    kwargs, schema = _tic_csv_kwargs(path, compact)
    started = time.perf_counter()
    df = pd.read_csv(path, **kwargs)
    # TIC releases and external stellar catalogs may call metallicity either MH
    # or feh. Keep the original column and expose MH as the canonical filter name.
    df = _normalize_catalog_columns(df)
    logger.info(
        "Loaded %d TIC rows and %d columns from %s in %.2fs (%s schema).",
        len(df), len(df.columns), path, time.perf_counter() - started,
        schema
    )
    return df


def iter_tic_csv(path: str, compact: bool, chunk_size: int):
    """Stream a large TIC CSV in bounded-memory DataFrame chunks."""
    kwargs, schema = _tic_csv_kwargs(path, compact)
    logger.info("Streaming %s in chunks of %d rows (%s schema).", path, chunk_size, schema)
    for chunk in pd.read_csv(path, chunksize=chunk_size, **kwargs):
        yield _normalize_catalog_columns(chunk)

# Metadata defining the 11 pre-sieve filters
FILTER_METADATA = {
    # Category 1: Target Purification & Noise Suppression
    "high_fidelity_photon": {
        "category": "1. Target Purification & Noise Suppression",
        "description": "High-Fidelity Photon Filter (Tmag <= 11.5)",
        "condition": "Tmag <= 11.5",
        "func": lambda df: df[df['Tmag'] <= 11.5]
    },
    "aperture_purity": {
        "category": "1. Target Purification & Noise Suppression",
        "description": "Aperture Purity Sieve (contratio <= 0.08)",
        "condition": "contratio <= 0.08",
        "func": lambda df: df[df['contratio'] <= 0.08]
    },
    "pure_dwarf": {
        "category": "1. Target Purification & Noise Suppression",
        "description": "Pure Main-Sequence Dwarf Sieve (logg >= 4.1 and rad <= 1.8)",
        "condition": "logg >= 4.1 and rad <= 1.8",
        "func": lambda df: df[(df['logg'] >= 4.1) & (df['rad'] <= 1.8)]
    },
    
    # Category 2: Target Selection by Spectral Class
    "ultra_cool_terrestrial": {
        "category": "2. Target Selection by Spectral Class",
        "description": "Ultra-Cool Terrestrial Sieve (Teff BETWEEN 2400 AND 3900, rad <= 0.6, logg >= 4.5)",
        "condition": "2400 <= Teff <= 3900, rad <= 0.6, logg >= 4.5",
        "func": lambda df: df[(df['Teff'] >= 2400) & (df['Teff'] <= 3900) & (df['rad'] <= 0.6) & (df['logg'] >= 4.5)]
    },
    "orange_dwarf": {
        "category": "2. Target Selection by Spectral Class",
        "description": "Orange Dwarf Sweet-Spot (Teff BETWEEN 3900 AND 5200, logg BETWEEN 4.2 AND 4.6)",
        "condition": "3900 <= Teff <= 5200, 4.2 <= logg <= 4.6",
        "func": lambda df: df[(df['Teff'] >= 3900) & (df['Teff'] <= 5200) & (df['logg'] >= 4.2) & (df['logg'] <= 4.6)]
    },
    "solar_twin": {
        "category": "2. Target Selection by Spectral Class",
        "description": "Solar Twin Catalog (Teff BETWEEN 5500 AND 6000, mass BETWEEN 0.9 AND 1.1, rad BETWEEN 0.9 AND 1.1)",
        "condition": "5500 <= Teff <= 6000, 0.9 <= mass <= 1.1, 0.9 <= rad <= 1.1",
        "func": lambda df: df[(df['Teff'] >= 5500) & (df['Teff'] <= 6000) & (df['mass'] >= 0.9) & (df['mass'] <= 1.1) & (df['rad'] >= 0.9) & (df['rad'] <= 1.1)]
    },
    
    # Category 3: Spatial & Temporal Orbit Maximization
    "cvz": {
        "category": "3. Spatial & Temporal Orbit Maximization",
        "description": "Continuous Viewing Zone Sieve (dec >= 78.0 OR dec <= -78.0)",
        "condition": "dec >= 78.0 or dec <= -78.0",
        "func": lambda df: df[(df['dec'] >= 78.0) | (df['dec'] <= -78.0)]
    },
    "neighborhood_census": {
        "category": "3. Spatial & Temporal Orbit Maximization",
        "description": "Neighborhood Census (dist <= 75.0 or plx >= 13.33)",
        "condition": "d <= 75.0 or plx >= 13.33",
        "func": lambda df: df[(df['d'] <= 75.0) | (df['plx'] >= 13.33)]
    },
    
    # Category 4: Exotic & Compositional Archetypes
    "gas_giant_nursery": {
        "category": "4. Exotic & Compositional Archetypes",
        "description": "Gas-Giant Nursery (feh >= 0.15 and Tmag <= 12.0)",
        "condition": "MH >= 0.15 and Tmag <= 12.0",
        "func": lambda df: df[(df['MH'] >= 0.15) & (df['Tmag'] <= 12.0)]
    },
    "primitive_system": {
        "category": "4. Exotic & Compositional Archetypes",
        "description": "Primitive System Archetype (feh <= -0.5)",
        "condition": "MH <= -0.5",
        "func": lambda df: df[df['MH'] <= -0.5]
    },
    "dead_star": {
        "category": "4. Exotic & Compositional Archetypes",
        "description": "Dead Star Transit Sieve (logg >= 7.0 and rad <= 0.03)",
        "condition": "logg >= 7.0 and rad <= 0.03",
        "func": lambda df: df[(df['logg'] >= 7.0) & (df['rad'] <= 0.03)]
    }
}


def chunk_list(lst: list, n: int):
    """Yield successive n-sized chunks from a list."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def fetch_observations_by_sector(sector: int, limit: int = None) -> list:
    """
    Query MAST TESS observations for a given TESS sector to get a list of observed TIC IDs.
    
    Parameters:
        sector (int): TESS Sector number (e.g., 1, 26).
        limit (int, optional): Maximum number of targets to return. Useful for quick testing.
        
    Returns:
        list: List of integer TIC IDs observed in the sector.
    """
    logger.info(f"Querying TESS observations for Sector {sector}...")
    try:
        _, Observations = _import_mast()
        obs_table = Observations.query_criteria(
            project="TESS",
            sequence_number=sector,
            provenance_name="TESS-SPOC"
        )
        
        if len(obs_table) == 0:
            logger.warning(f"No SPOC observations found for Sector {sector}. Trying QLP...")
            obs_table = Observations.query_criteria(
                project="TESS",
                sequence_number=sector,
                provenance_name="QLP"
            )
            
        logger.info(f"Retrieved {len(obs_table)} observations from MAST.")
        
        # Clean and extract unique target names (which are TIC IDs)
        target_names = obs_table['target_name']
        tic_ids = []
        for target in target_names:
            try:
                # Strip prefix if it exists (e.g., 'TIC 1234' -> '1234')
                clean_id = str(target).replace('TIC', '').strip()
                if clean_id.isdigit():
                    tic_ids.append(int(clean_id))
            except ValueError:
                continue
                
        # Drop duplicates while keeping order
        unique_tic_ids = list(dict.fromkeys(tic_ids))
        logger.info(f"Extracted {len(unique_tic_ids)} unique TIC IDs.")
        
        if limit is not None:
            logger.info(f"Applying target limit constraint: slicing to first {limit} targets.")
            unique_tic_ids = unique_tic_ids[:limit]
            
        return unique_tic_ids
        
    except Exception as e:
        logger.error(f"Error querying MAST observations: {e}")
        raise e


def fetch_tic_properties(tic_ids: list, chunk_size: int = 1000) -> pd.DataFrame:
    """
    Query the MAST TIC catalog for a list of TIC IDs in chunks.
    
    Parameters:
        tic_ids (list): List of integer TIC IDs.
        chunk_size (int): Size of chunks to submit in single API queries.
        
    Returns:
        pd.DataFrame: DataFrame containing TIC stellar parameters.
    """
    logger.info(f"Querying TIC catalog for stellar properties of {len(tic_ids)} targets in chunks of {chunk_size}...")
    Catalogs, _ = _import_mast()
    dfs = []
    
    for i, chunk in enumerate(chunk_list(tic_ids, chunk_size)):
        t_start = time.time()
        try:
            result = Catalogs.query_criteria(catalog="Tic", ID=chunk)
            if len(result) > 0:
                dfs.append(result.to_pandas())
                logger.info(f"Chunk {i+1}: Retrieved {len(result)} records in {time.time() - t_start:.2f}s.")
            else:
                logger.warning(f"Chunk {i+1}: Returned 0 records.")
        except Exception as e:
            logger.error(f"Error querying TIC catalog in chunk {i+1}: {e}")
            
    if not dfs:
        logger.warning("No data retrieved from TIC catalog.")
        return pd.DataFrame()
        
    full_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Completed TIC catalog query. Total rows: {len(full_df)}")
    return full_df


def fetch_gaia_ruwe(gaia_ids: list, chunk_size: int = 1000) -> pd.DataFrame:
    """
    Query the Gaia DR3 archive to retrieve RUWE values for a list of Gaia source IDs.
    
    Parameters:
        gaia_ids (list): List of Gaia source ID values.
        chunk_size (int): Chunk size for ADQL batch querying.
        
    Returns:
        pd.DataFrame: DataFrame with columns 'source_id' and 'ruwe'.
    """
    Gaia = _import_gaia()
    valid_ids = []
    for gid in gaia_ids:
        if pd.isna(gid):
            continue
        try:
            # Convert to string, strip decimal parts if it's represented as a float string, and check digit status
            clean_str = str(gid).split('.')[0].strip()
            if clean_str.isdigit():
                val_int = int(clean_str)
                if val_int > 0:
                    valid_ids.append(val_int)
        except (ValueError, TypeError):
            continue

    if not valid_ids:
        logger.warning("No valid Gaia IDs found in TIC dataset to fetch RUWE.")
        return pd.DataFrame(columns=['source_id', 'ruwe'])
        
    logger.info(f"Querying Gaia DR3 for RUWE of {len(valid_ids)} targets in chunks of {chunk_size}...")
    dfs = []
    
    for i, chunk in enumerate(chunk_list(valid_ids, chunk_size)):
        t_start = time.time()
        try:
            id_list_str = ",".join(map(str, chunk))
            query = f"SELECT source_id, ruwe FROM gaiadr3.gaia_source WHERE source_id IN ({id_list_str})"
            job = Gaia.launch_job(query)
            result_df = job.get_results().to_pandas()
            if len(result_df) > 0:
                dfs.append(result_df)
                logger.info(f"Gaia Chunk {i+1}: Retrieved {len(result_df)} records in {time.time() - t_start:.2f}s.")
            else:
                logger.warning(f"Gaia Chunk {i+1}: Returned 0 records.")
        except Exception as e:
            logger.error(f"Error querying Gaia database in chunk {i+1}: {e}")
            
    if not dfs:
        logger.warning("No Gaia RUWE data retrieved.")
        return pd.DataFrame(columns=['source_id', 'ruwe'])
        
    full_gaia_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Completed Gaia RUWE query. Total rows: {len(full_gaia_df)}")
    return full_gaia_df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset using the strict sequential 4-step pipeline:
    1. Object Type Clean: Drop all galaxies and non-stellar objects (objType == 'STAR').
    2. Artifact Purge: Drop duplicates and phantom sources by verifying the disposition column.
    3. Contamination Cut: Drop blended targets to prevent false positives (contratio <= 0.1).
    4. Astrometric Clean: Filter out binary stars using Renormalised Unit Weight Error (ruwe <= 1.4).
    
    Parameters:
        df (pd.DataFrame): Input merged dataset.
        
    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    logger.info("========================================")
    logger.info("   Executing Dataset Cleaning Pipeline")
    logger.info("========================================")
    initial_count = len(df)
    
    # Step 1: Object Type Clean
    if 'objType' in df.columns:
        count_before = len(df)
        df = df[df['objType'].astype(str).str.upper() == 'STAR']
        logger.info(f"Step 1 (Object Type Clean): objType == 'STAR' -> Retained {len(df)} / {count_before} targets.")
    else:
        logger.warning("Step 1 (Object Type Clean) Skipped: 'objType' column missing.")
        
    # Step 2: Artifact Purge
    if 'disposition' in df.columns:
        count_before = len(df)
        # Drop duplicates/artifacts. Valid dispositions are NaN/NULL.
        # We drop if it matches DUPLICATE, ARTIFACT, or code values '6', '7'.
        disp_series = df['disposition'].astype(str).str.upper().str.strip()
        df = df[~disp_series.isin(['DUPLICATE', 'ARTIFACT', '6.0', '7.0', '6', '7'])]
        logger.info(f"Step 2 (Artifact Purge): Disposition Cleaned -> Retained {len(df)} / {count_before} targets.")
    else:
        logger.warning("Step 2 (Artifact Purge) Skipped: 'disposition' column missing.")
        
    # Step 3: Contamination Cut
    if 'contratio' in df.columns:
        count_before = len(df)
        # Filter contratio <= 0.1. Note that NaNs will be dropped.
        df = df[df['contratio'] <= 0.1]
        logger.info(f"Step 3 (Contamination Cut): contratio <= 0.1 -> Retained {len(df)} / {count_before} targets.")
    else:
        logger.warning("Step 3 (Contamination Cut) Skipped: 'contratio' column missing.")
        
    # Step 4: Astrometric Clean
    if 'ruwe' in df.columns:
        count_before = len(df)
        # Filter ruwe <= 1.4. Note that NaNs will be dropped.
        df = df[df['ruwe'] <= 1.4]
        logger.info(f"Step 4 (Astrometric Clean): ruwe <= 1.4 -> Retained {len(df)} / {count_before} targets.")
    else:
        logger.warning("Step 4 (Astrometric Clean) Skipped: 'ruwe' column missing.")
        
    final_count = len(df)
    logger.info(f"Cleaning completed! Total targets remaining: {final_count} / {initial_count} (Dropped {initial_count - final_count}).")
    return df


def impute_stellar_parameters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing radius (rad) or mass (mass) values using physical scaling relations:
    1. Stefan-Boltzmann scaling for radius (if Teff and luminosity are available).
    2. Gravitational scaling for radius (if mass and logg are available).
    3. Gravitational scaling for mass (if radius and logg are available).
    4. Piecewise Main-Sequence Mass-Radius relation fallback.
    
    Parameters:
        df (pd.DataFrame): Dataset to impute parameters on.
        
    Returns:
        pd.DataFrame: Imputed dataset.
    """
    logger.info("========================================")
    logger.info("  Executing Physical Parameter Imputation")
    logger.info("========================================")
    
    df = df.copy()
    
    # 1. Stefan-Boltzmann Radius Imputation
    # Relation: R/R_sun = sqrt(L/L_sun) * (T_sun / T_eff)^2
    # Where T_sun = 5778 K
    if 'rad' in df.columns and 'lum' in df.columns and 'Teff' in df.columns:
        sb_cond = df['rad'].isna() & df['lum'].notna() & df['Teff'].notna() & (df['Teff'] > 0)
        if sb_cond.any():
            imputed_rad = np.sqrt(df.loc[sb_cond, 'lum']) * (5778.0 / df.loc[sb_cond, 'Teff'])**2
            df.loc[sb_cond, 'rad'] = imputed_rad
            logger.info(f"Imputed {sb_cond.sum()} radius values using Stefan-Boltzmann scaling (Teff & luminosity).")
            
    # 2. Gravitational Radius Imputation
    # Relation: R/R_sun = sqrt( (M/M_sun) / 10^(logg - logg_sun) )
    # Where logg_sun = 4.4378 dex
    if 'rad' in df.columns and 'mass' in df.columns and 'logg' in df.columns:
        grav_cond = df['rad'].isna() & df['mass'].notna() & df['logg'].notna()
        if grav_cond.any():
            imputed_rad = np.sqrt(df.loc[grav_cond, 'mass'] / (10**(df.loc[grav_cond, 'logg'] - 4.4378)))
            df.loc[grav_cond, 'rad'] = imputed_rad
            logger.info(f"Imputed {grav_cond.sum()} radius values using gravitational scaling (mass & logg).")
            
    # 3. Gravitational Mass Imputation
    # Relation: M/M_sun = (R/R_sun)^2 * 10^(logg - logg_sun)
    # Where logg_sun = 4.4378 dex
    if 'mass' in df.columns and 'rad' in df.columns and 'logg' in df.columns:
        grav_mass_cond = df['mass'].isna() & df['rad'].notna() & df['logg'].notna()
        if grav_mass_cond.any():
            imputed_mass = (df.loc[grav_mass_cond, 'rad']**2) * (10**(df.loc[grav_mass_cond, 'logg'] - 4.4378))
            df.loc[grav_mass_cond, 'mass'] = imputed_mass
            logger.info(f"Imputed {grav_mass_cond.sum()} mass values using gravitational scaling (radius & logg).")
            
    # 4. Empirical Main-Sequence Mass-Radius Relation Fallback
    # Piecewise relation:
    #   For R < 1.0 R_sun: M = R^(1.25)  (assuming R ~ M^0.8)
    #   For R >= 1.0 R_sun: M = R^(1.75) (assuming R ~ M^0.57)
    if 'mass' in df.columns and 'rad' in df.columns:
        emp_cond = df['mass'].isna() & df['rad'].notna()
        if emp_cond.any():
            # Apply piecewise mapping
            radii = df.loc[emp_cond, 'rad']
            imputed_mass = np.where(radii < 1.0, radii ** 1.25, radii ** 1.75)
            df.loc[emp_cond, 'mass'] = imputed_mass
            logger.info(f"Imputed {emp_cond.sum()} mass values using Main-Sequence empirical relations.")
            
    return df


def apply_filters(df: pd.DataFrame, enabled_filters: list) -> pd.DataFrame:
    """
    Apply a selected suite of pre-sieve filters to the cleaned dataset.
    
    Parameters:
        df (pd.DataFrame): The cleaned and imputed dataset.
        enabled_filters (list): List of filter keys to apply.
        
    Returns:
        pd.DataFrame: Filtered dataset.
    """
    logger.info("========================================")
    logger.info("   Applying Astrophysical Filters")
    logger.info("========================================")
    
    filtered_df = df.copy()
    
    for fkey in enabled_filters:
        if fkey not in FILTER_METADATA:
            logger.warning(f"Filter key '{fkey}' is invalid and will be skipped.")
            continue
            
        metadata = FILTER_METADATA[fkey]
        count_before = len(filtered_df)
        
        # Apply the lambda filter function
        filtered_df = metadata["func"](filtered_df)
        
        logger.info(f"Filter [{fkey}]: {metadata['description']}")
        logger.info(f"  -> Retained {len(filtered_df)} / {count_before} targets (Dropped {count_before - len(filtered_df)}).")
        
    return filtered_df


def categorize_targets(df: pd.DataFrame, enabled_filters: list) -> pd.DataFrame:
    """Add independent filter flags and labels without intersecting categories."""
    result = df.copy()
    flag_columns = []
    for fkey in enabled_filters:
        if fkey not in FILTER_METADATA:
            logger.warning("Filter key '%s' is invalid and will be skipped.", fkey)
            continue
        flag = f"is_{fkey}"
        # Reuse the canonical filter and align its retained indices to the input.
        retained_index = FILTER_METADATA[fkey]['func'](df).index
        result[flag] = result.index.isin(retained_index)
        flag_columns.append(flag)
        logger.info("Category [%s]: matched %d targets.", fkey, result[flag].sum())

    if flag_columns:
        flags = result[flag_columns].to_numpy(dtype=bool)
        names = np.asarray([name.removeprefix('is_') for name in flag_columns])
        result['selection_count'] = flags.sum(axis=1)
        result['selection_labels'] = [','.join(names[row]) for row in flags]
    return result


def process_dataframe(df: pd.DataFrame, filters: list, categorize: bool) -> pd.DataFrame:
    """Run the common cleaning, imputation, and selection stages on one frame."""
    prepared = impute_stellar_parameters(clean_dataset(df))
    if categorize:
        return categorize_targets(prepared, filters)
    if filters:
        return apply_filters(prepared, filters)
    return prepared


def process_local_csv(path: str, output: str, compact: bool, chunk_size: int,
                      filters: list, categorize: bool) -> tuple[int, int]:
    """Process and export a large local catalog incrementally."""
    input_rows = output_rows = 0
    category_totals = {f"is_{name}": 0 for name in filters} if categorize else {}
    wrote_header = False
    started = time.perf_counter()

    logger.info("============================================================")
    logger.info("TESS CATALOG PRE-SIEVE")
    logger.info("Input:  %s", path)
    logger.info("Output: %s", output)
    logger.info("Mode:   %s", "complete catalog columns + analysis" if not compact else "compact analysis columns")
    logger.info("Each chunk is cleaned first, then categorized independently.")
    logger.info("A category match is a label; categories are not intersected.")
    logger.info("============================================================")

    for chunk_number, chunk in enumerate(iter_tic_csv(path, compact, chunk_size), 1):
        input_rows += len(chunk)
        if chunk_number == 1:
            logger.info("Cleaning rules active:")
            logger.info("  1. Keep rows where objType is STAR")
            logger.info("  2. Remove rows flagged as duplicate or artifact")
            logger.info("  3. Require contamination ratio <= 0.10")
            if 'ruwe' in chunk.columns:
                logger.info("  4. Require RUWE <= 1.40")
            else:
                logger.info("  4. RUWE check skipped: this input has no RUWE column")
        # Detailed per-filter messages are useful for debugging but overwhelming
        # for multi-million-row files. The summary below reports the same outcome.
        previous_level = logger.level
        logger.setLevel(logging.ERROR)
        try:
            result = process_dataframe(chunk, filters, categorize)
        finally:
            logger.setLevel(previous_level)
        output_rows += len(result)
        for column in category_totals:
            if column in result:
                category_totals[column] += int(result[column].sum())
        result.to_csv(output, mode='w' if not wrote_header else 'a',
                      header=not wrote_header, index=False)
        wrote_header = True
        if chunk_number == 1 or chunk_number % 10 == 0:
            logger.info(
                "Progress | rows examined: %12s | passed cleaning: %10s",
                f"{input_rows:,}", f"{output_rows:,}"
            )
    if not wrote_header:
        pd.DataFrame().to_csv(output, index=False)

    elapsed = time.perf_counter() - started
    logger.info("============================================================")
    logger.info("ANALYSIS COMPLETE")
    logger.info("Rows examined:          %s", f"{input_rows:,}")
    logger.info("Rows passing cleaning:  %s", f"{output_rows:,}")
    logger.info("Rows rejected:          %s", f"{input_rows - output_rows:,}")
    logger.info("Runtime:                %.2f seconds", elapsed)
    if category_totals:
        logger.info("Category matches (one star may match several):")
        for column, count in sorted(category_totals.items(), key=lambda item: item[1], reverse=True):
            logger.info("  %-30s %s", column.removeprefix('is_'), f"{count:,}")
    logger.info("Saved analyzed CSV: %s", output)
    logger.info("============================================================")
    return input_rows, output_rows


def list_available_filters():
    """Print all available filters grouped by category."""
    print("\nAvailable Pre-Sieve Filters:")
    print("====================================================================================================")
    current_cat = ""
    for fkey, val in FILTER_METADATA.items():
        if val["category"] != current_cat:
            current_cat = val["category"]
            print(f"\n{current_cat}:")
            print("-" * 50)
        print(f"  - {fkey:<25} | {val['description']}")
    print("====================================================================================================")


def main():
    parser = argparse.ArgumentParser(
        description="Query, Clean, Impute, and Filter TESS Input Catalog (TIC) Targets.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--sector", type=int, default=1, help="TESS Sector number to query (default: 1)")
    parser.add_argument("--input-csv", help="Process an existing headered or 125-column headerless TIC CSV; skips MAST/Gaia network queries")
    parser.add_argument("--compact", action="store_true", help="Output only columns needed for cleaning and pre-sieve selection (fastest)")
    parser.add_argument("--complete-output", action="store_true", help="Preserve all original CSV columns and append analysis columns")
    parser.add_argument("--csv-chunk-size", type=int, default=250000, help="Rows per chunk for local CSV streaming (default: 250000)")
    parser.add_argument("--limit", type=int, default=1000, help="Limit number of query targets to prevent API timeouts (default: 1000). Set to 0 for unlimited.")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for MAST and Gaia queries (default: 1000)")
    parser.add_argument("--filters", type=str, help="Comma-separated list of filter keys to apply.\nExample: --filters solar_twin,neighborhood_census")
    parser.add_argument("--all-filters", action="store_true", help="Apply all 11 filters sequentially (intersection)")
    parser.add_argument("--categorize", action="store_true", help="Add independent boolean flags and labels for selected filters instead of intersecting them")
    parser.add_argument("--categorize-all", action="store_true", help="Categorize independently with all 11 pre-sieve filters")
    parser.add_argument("--list-filters", action="store_true", help="List all available filters and exit")
    parser.add_argument("--output", type=str, help="Output CSV path; local inputs default to <input>_analyzed.csv beside the input file")
    
    args = parser.parse_args()
    
    if args.list_filters:
        list_available_filters()
        sys.exit(0)

    if args.compact and args.complete_output:
        parser.error("Choose either --compact or --complete-output, not both.")

    if args.output is None:
        if args.input_csv:
            source = Path(args.input_csv)
            args.output = str(source.with_name(f"{source.stem}_analyzed.csv"))
        else:
            args.output = "filtered_tic_targets.csv"
        
    t_pipeline_start = time.time()

    filters_to_apply = []
    if args.all_filters or args.categorize_all:
        filters_to_apply = list(FILTER_METADATA.keys())
    elif args.filters:
        filters_to_apply = [f.strip() for f in args.filters.split(",") if f.strip()]
    
    if args.input_csv:
        try:
            process_local_csv(
                args.input_csv, args.output, args.compact and not args.complete_output, args.csv_chunk_size,
                filters_to_apply, args.categorize or args.categorize_all
            )
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            logger.error("Could not read TIC CSV: %s", exc)
            sys.exit(1)
        logger.info("Pipeline executed successfully in %.2f seconds!", time.time() - t_pipeline_start)
        return
    else:
        # Remote acquisition path retained for users who do not yet have a TIC CSV.
        limit_val = args.limit if args.limit > 0 else None
        try:
            tic_ids = fetch_observations_by_sector(args.sector, limit=limit_val)
            if not tic_ids:
                raise RuntimeError(f"No TIC targets found for Sector {args.sector}.")
            df_tic = fetch_tic_properties(tic_ids, chunk_size=args.chunk_size)
            if df_tic.empty:
                raise RuntimeError("No TIC catalog data retrieved.")
            df_gaia = fetch_gaia_ruwe(df_tic['GAIA'].dropna().unique().tolist(), args.chunk_size)
        except Exception as exc:
            logger.error("Fatal remote query error: %s", exc)
            sys.exit(1)

        logger.info("Merging TIC catalog data with Gaia RUWE measurements...")
        # Strings preserve all 64 bits of Gaia source IDs; float conversion does not.
        df_tic['GAIA'] = df_tic['GAIA'].astype('string')
        df_gaia['source_id'] = df_gaia['source_id'].astype('string')
        merged_df = df_tic.merge(df_gaia, left_on='GAIA', right_on='source_id', how='left')
    
    # 5. Clean Dataset
    cleaned_df = clean_dataset(merged_df)
    
    # 6. Physical Parameter Imputation
    imputed_df = impute_stellar_parameters(cleaned_df)
    
    # 7. Apply Astrophysical Filters
    if args.categorize or args.categorize_all:
        final_df = categorize_targets(imputed_df, filters_to_apply)
    elif filters_to_apply:
        final_df = apply_filters(imputed_df, filters_to_apply)
    else:
        logger.info("No filters specified. Skipping step. Exporting cleaned & imputed dataset.")
        final_df = imputed_df
        
    # 8. Export Results
    logger.info(f"Exporting final filtered dataset containing {len(final_df)} targets to {args.output}...")
    try:
        final_df.to_csv(args.output, index=False)
        logger.info(f"Pipeline executed successfully in {time.time() - t_pipeline_start:.2f} seconds!")
    except Exception as e:
        logger.error(f"Error exporting CSV file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
