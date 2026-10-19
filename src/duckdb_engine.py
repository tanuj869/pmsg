import os
import duckdb
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "pmsg.duckdb")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

def get_connection(db_path: str = DB_PATH) -> duckdb.DuckDBPyConnection:
    """Returns a DuckDB connection to the persistent or in-memory analytics warehouse."""
    conn = duckdb.connect(database=db_path)
    return conn

def init_duckdb_views(conn: duckdb.DuckDBPyConnection = None) -> duckdb.DuckDBPyConnection:
    """Registers analytical views directly over the processed CSVs in DuckDB."""
    if conn is None:
        conn = get_connection()

    tables = {
        "snapshots_long": os.path.join(PROCESSED_DIR, "pmsg_snapshots_long.csv"),
        "covariates_wide": os.path.join(PROCESSED_DIR, "state_covariates_wide.csv"),
        "modeling_table": os.path.join(PROCESSED_DIR, "modeling_table.csv"),
        "tariff_shock": os.path.join(PROCESSED_DIR, "scenario_d_tariff_shock.csv"),
        "national_series": os.path.join(PROCESSED_DIR, "national_installations_series.csv"),
        "targeting_states": os.path.join(PROCESSED_DIR, "scenario_c_targeting.csv"),
    }

    for view_name, file_path in tables.items():
        if os.path.exists(file_path):
            norm_path = file_path.replace("\\", "/")
            conn.execute(f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM read_csv_auto('{norm_path}');")

    return conn

def execute_sql(sql: str) -> pd.DataFrame:
    """Executes arbitrary SQL query using DuckDB and returns a Pandas DataFrame."""
    conn = get_connection()
    init_duckdb_views(conn)
    df = conn.execute(sql).df()
    conn.close()
    return df

def get_top_states_by_conversion(limit: int = 10) -> pd.DataFrame:
    """Returns top N states ranked by application-to-installation conversion ratio."""
    sql = f"""
        SELECT 
            state, 
            ROUND(target_conversion_ratio * 100, 2) AS conversion_ratio_pct,
            applications_aug2026 AS applications,
            installations_aug2026 AS installations,
            ROUND(irradiance_annual_kwh_m2_day, 2) AS irradiance,
            tariff_domestic4kw_2025_paise_per_kwh AS tariff_paise
        FROM modeling_table
        ORDER BY target_conversion_ratio DESC
        LIMIT {limit};
    """
    return execute_sql(sql)

def get_state_metrics(state_name: str) -> dict:
    """Returns detailed demographic, adoption, and resource metrics for a single state."""
    sql = f"""
        SELECT *
        FROM modeling_table
        WHERE LOWER(state) = LOWER('{state_name}')
        LIMIT 1;
    """
    df = execute_sql(sql)
    if df.empty:
        return {}
    return df.iloc[0].to_dict()

def get_national_summary() -> dict:
    """Aggregates national adoption totals via DuckDB SQL."""
    sql = """
        SELECT 
            COUNT(DISTINCT state) AS total_states,
            SUM(applications_aug2026) AS total_applications,
            SUM(installations_aug2026) AS total_installations,
            ROUND(AVG(target_conversion_ratio) * 100, 2) AS avg_conversion_ratio_pct,
            MIN(target_conversion_ratio) * 100 AS min_conversion_ratio_pct,
            MAX(target_conversion_ratio) * 100 AS max_conversion_ratio_pct
        FROM modeling_table;
    """
    df = execute_sql(sql)
    return df.iloc[0].to_dict()

if __name__ == "__main__":
    print("Initializing DuckDB Engine & registering views...")
    conn = init_duckdb_views()
    print("Tables registered successfully in DuckDB.")
    
    print("\n--- National Summary (via DuckDB SQL) ---")
    summary = get_national_summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print("\n--- Top 5 States by Conversion Ratio (via DuckDB SQL) ---")
    top5 = get_top_states_by_conversion(5)
    print(top5)
    conn.close()
