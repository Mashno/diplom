from __future__ import annotations

from datetime import date
from itertools import islice
from typing import Any, Iterable, Iterator, Sequence

from psycopg import Connection
from psycopg.rows import dict_row

from config import (
    CMAPSS_OPERATIONAL_COLUMNS,
    CMAPSS_SENSOR_COLUMNS,
    DEFAULT_CMAPSS_DATASET_ID,
    get_cmapss_dataset_prefix,
    get_cmapss_source_system,
)


def _chunked(iterable: Iterable[tuple[Any, ...]], chunk_size: int) -> Iterator[list[tuple[Any, ...]]]:
    """Yield tuples from an iterable in fixed-size chunks."""

    iterator = iter(iterable)
    while True:
        chunk = list(islice(iterator, chunk_size))
        if not chunk:
            return
        yield chunk


def _equipment_code(dataset_id: str, unit_number: int) -> str:
    """Build a stable equipment code for a C-MAPSS engine unit."""

    return f"CMAPSS-{dataset_id.upper()}-UNIT-{unit_number:03d}"


def _equipment_name(dataset_id: str, unit_number: int) -> str:
    """Build a readable equipment name for a C-MAPSS engine unit."""

    return f"C-MAPSS {dataset_id.upper()} Engine Unit {unit_number:03d}"


def _parameter_name(parameter_code: str) -> str:
    """Convert a telemetry parameter code into a readable name."""

    prefix, suffix = parameter_code.rsplit("_", maxsplit=1)
    label_prefix = "Operational setting" if prefix == "operational_setting" else "Sensor"
    return f"{label_prefix} {suffix}"


def get_equipment(
    connection: Connection,
    dataset_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return equipment records, optionally filtered to a C-MAPSS dataset."""

    if dataset_id:
        query = """
            select
                equipment_id,
                equipment_code,
                equipment_name,
                equipment_type,
                workshop,
                location,
                criticality_level,
                status
            from app.equipment
            where equipment_code like %s
            order by equipment_code
        """
        params = (f"{get_cmapss_dataset_prefix(dataset_id)}%",)
    else:
        query = """
            select
                equipment_id,
                equipment_code,
                equipment_name,
                equipment_type,
                workshop,
                location,
                criticality_level,
                status
            from app.equipment
            where status = 'in_operation'
            order by equipment_code
        """
        params = ()

    with connection.cursor(row_factory=dict_row) as cursor:
        cursor.execute(query, params)
        return list(cursor.fetchall())


def upsert_cmapss_equipment(
    connection: Connection,
    unit_numbers: Sequence[int],
    dataset_id: str = DEFAULT_CMAPSS_DATASET_ID,
) -> dict[int, dict[str, Any]]:
    """Create or update equipment records for C-MAPSS engine units."""

    unique_units = sorted({int(unit_number) for unit_number in unit_numbers})
    if not unique_units:
        return {}

    payload = [
        (
            _equipment_code(dataset_id, unit_number),
            _equipment_name(dataset_id, unit_number),
            "Turbofan engine",
            "NASA C-MAPSS dataset",
            dataset_id.upper(),
            "high",
            "in_operation",
        )
        for unit_number in unique_units
    ]

    query = """
        insert into app.equipment (
            equipment_code,
            equipment_name,
            equipment_type,
            workshop,
            location,
            criticality_level,
            status
        )
        values (%s, %s, %s, %s, %s, %s, %s)
        on conflict (equipment_code) do update
        set
            equipment_name = excluded.equipment_name,
            equipment_type = excluded.equipment_type,
            workshop = excluded.workshop,
            location = excluded.location,
            criticality_level = excluded.criticality_level,
            status = excluded.status
    """
    try:
        with connection.cursor() as cursor:
            cursor.executemany(query, payload)
        connection.commit()
    except Exception:
        connection.rollback()
        raise

    code_to_unit = {_equipment_code(dataset_id, unit_number): unit_number for unit_number in unique_units}
    fetch_query = """
        select
            equipment_id,
            equipment_code,
            equipment_name,
            equipment_type,
            workshop,
            location,
            criticality_level,
            status
        from app.equipment
        where equipment_code = any(%s)
        order by equipment_code
    """
    with connection.cursor(row_factory=dict_row) as cursor:
        cursor.execute(fetch_query, (list(code_to_unit.keys()),))
        records = list(cursor.fetchall())

    return {code_to_unit[record["equipment_code"]]: dict(record) for record in records}


def upsert_cmapss_telemetry_parameters(connection: Connection) -> dict[str, int]:
    """Create or update telemetry parameter records for C-MAPSS columns."""

    parameter_codes = [*CMAPSS_OPERATIONAL_COLUMNS, *CMAPSS_SENSOR_COLUMNS]
    payload = [
        (
            parameter_code,
            _parameter_name(parameter_code),
            "relative",
            None,
            None,
            None,
            None,
            f"C-MAPSS parameter {parameter_code}",
        )
        for parameter_code in parameter_codes
    ]

    query = """
        insert into telemetry.telemetry_parameters (
            parameter_code,
            parameter_name,
            unit,
            normal_min,
            normal_max,
            critical_min,
            critical_max,
            description
        )
        values (%s, %s, %s, %s, %s, %s, %s, %s)
        on conflict (parameter_code) do update
        set
            parameter_name = excluded.parameter_name,
            unit = excluded.unit,
            normal_min = excluded.normal_min,
            normal_max = excluded.normal_max,
            critical_min = excluded.critical_min,
            critical_max = excluded.critical_max,
            description = excluded.description
    """
    try:
        with connection.cursor() as cursor:
            cursor.executemany(query, payload)
        connection.commit()
    except Exception:
        connection.rollback()
        raise

    return get_parameter_id_map(connection, parameter_codes=parameter_codes)


def get_parameter_id_map(
    connection: Connection,
    parameter_codes: Sequence[str] | None = None,
) -> dict[str, int]:
    """Return a mapping from parameter_code to parameter_id."""

    if parameter_codes:
        query = """
            select parameter_id, parameter_code
            from telemetry.telemetry_parameters
            where parameter_code = any(%s)
            order by parameter_code
        """
        params = (list(parameter_codes),)
    else:
        query = """
            select parameter_id, parameter_code
            from telemetry.telemetry_parameters
            order by parameter_code
        """
        params = ()

    with connection.cursor() as cursor:
        cursor.execute(query, params)
        return {parameter_code: int(parameter_id) for parameter_id, parameter_code in cursor.fetchall()}


def delete_previous_cmapss_measurements(
    connection: Connection,
    dataset_id: str = DEFAULT_CMAPSS_DATASET_ID,
) -> int:
    """Delete previously loaded C-MAPSS telemetry rows for one dataset."""

    query = """
        delete from telemetry.telemetry_measurements
        where source_system = %s
    """
    try:
        with connection.cursor() as cursor:
            cursor.execute(query, (get_cmapss_source_system(dataset_id),))
            deleted_rows = cursor.rowcount
        connection.commit()
    except Exception:
        connection.rollback()
        raise

    return deleted_rows


def get_cmapss_measurement_count(
    connection: Connection,
    dataset_id: str = DEFAULT_CMAPSS_DATASET_ID,
) -> int:
    """Return the number of C-MAPSS telemetry rows stored for one dataset."""

    query = """
        select count(*)
        from telemetry.telemetry_measurements
        where source_system = %s
    """
    with connection.cursor() as cursor:
        cursor.execute(query, (get_cmapss_source_system(dataset_id),))
        return int(cursor.fetchone()[0])


def insert_telemetry_measurements(
    connection: Connection,
    rows: Iterable[tuple[Any, ...]],
    chunk_size: int = 5_000,
) -> int:
    """Insert telemetry rows in batches using executemany."""

    query = """
        insert into telemetry.telemetry_measurements (
            measured_at,
            equipment_id,
            parameter_id,
            quality_code_id,
            value,
            source_system
        )
        values (%s, %s, %s, %s, %s, %s)
    """
    inserted_rows = 0
    for chunk in _chunked(rows, chunk_size):
        try:
            with connection.cursor() as cursor:
                cursor.executemany(query, chunk)
            connection.commit()
            inserted_rows += len(chunk)
        except Exception:
            connection.rollback()
            raise

    return inserted_rows


def insert_diagnostic_report(
    connection: Connection,
    equipment_id: int,
    technical_state: str,
    degradation_index: float,
    anomaly_score: float,
    summary: str,
) -> dict[str, Any]:
    """Persist a diagnostic report and return the inserted row."""

    query = """
        insert into app.diagnostic_reports (
            equipment_id,
            technical_state,
            degradation_index,
            anomaly_score,
            summary
        )
        values (%s, %s, %s, %s, %s)
        returning
            diagnostic_report_id,
            equipment_id,
            report_time,
            technical_state,
            degradation_index::double precision as degradation_index,
            anomaly_score::double precision as anomaly_score,
            summary
    """
    try:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                query,
                (
                    equipment_id,
                    technical_state,
                    float(degradation_index),
                    float(anomaly_score),
                    summary,
                ),
            )
            record = cursor.fetchone()
        connection.commit()
    except Exception:
        connection.rollback()
        raise

    return dict(record)


def insert_rul_forecast(
    connection: Connection,
    equipment_id: int,
    diagnostic_report_id: int | None,
    predicted_rul_hours: float,
    lower_bound_hours: float,
    upper_bound_hours: float,
    confidence_score: float,
    model_name: str,
    model_version: str,
) -> dict[str, Any]:
    """Persist an RUL forecast and return the inserted row."""

    query = """
        insert into app.rul_forecasts (
            equipment_id,
            diagnostic_report_id,
            predicted_rul_hours,
            lower_bound_hours,
            upper_bound_hours,
            confidence_score,
            model_name,
            model_version
        )
        values (%s, %s, %s, %s, %s, %s, %s, %s)
        returning
            rul_forecast_id,
            equipment_id,
            diagnostic_report_id,
            forecast_time,
            predicted_rul_hours::double precision as predicted_rul_hours,
            lower_bound_hours::double precision as lower_bound_hours,
            upper_bound_hours::double precision as upper_bound_hours,
            confidence_score::double precision as confidence_score,
            model_name,
            model_version
    """
    try:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                query,
                (
                    equipment_id,
                    diagnostic_report_id,
                    float(predicted_rul_hours),
                    float(lower_bound_hours),
                    float(upper_bound_hours),
                    float(confidence_score),
                    model_name,
                    model_version,
                ),
            )
            record = cursor.fetchone()
        connection.commit()
    except Exception:
        connection.rollback()
        raise

    return dict(record)


def insert_maintenance_recommendation(
    connection: Connection,
    equipment_id: int,
    rul_forecast_id: int | None,
    diagnostic_report_id: int | None,
    recommended_work_id: int | None,
    priority: str,
    recommendation_text: str,
    recommended_start_date: date | None,
    recommended_deadline: date | None,
) -> dict[str, Any]:
    """Persist a maintenance recommendation and return the inserted row."""

    query = """
        insert into app.maintenance_recommendations (
            equipment_id,
            rul_forecast_id,
            diagnostic_report_id,
            recommended_work_id,
            priority,
            recommendation_text,
            recommended_start_date,
            recommended_deadline
        )
        values (%s, %s, %s, %s, %s, %s, %s, %s)
        returning
            recommendation_id,
            equipment_id,
            rul_forecast_id,
            diagnostic_report_id,
            recommended_work_id,
            priority,
            recommendation_text,
            recommended_start_date,
            recommended_deadline
    """
    try:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                query,
                (
                    equipment_id,
                    rul_forecast_id,
                    diagnostic_report_id,
                    recommended_work_id,
                    priority,
                    recommendation_text,
                    recommended_start_date,
                    recommended_deadline,
                ),
            )
            record = cursor.fetchone()
        connection.commit()
    except Exception:
        connection.rollback()
        raise

    return dict(record)


def get_maintenance_work_id_by_code(
    connection: Connection,
    work_code: str,
) -> int | None:
    """Return work_id for a maintenance work code."""

    query = """
        select work_id
        from app.maintenance_works
        where work_code = %s
        limit 1
    """
    with connection.cursor() as cursor:
        cursor.execute(query, (work_code,))
        row = cursor.fetchone()
    return int(row[0]) if row else None


def get_latest_results(
    connection: Connection,
    dataset_id: str = DEFAULT_CMAPSS_DATASET_ID,
    equipment_codes: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Return the latest diagnostic, RUL and recommendation for C-MAPSS equipment."""

    where_clause = "e.equipment_code = any(%s)" if equipment_codes else "e.equipment_code like %s"
    where_params: tuple[Any, ...]
    if equipment_codes:
        where_params = (list(equipment_codes),)
    else:
        where_params = (f"{get_cmapss_dataset_prefix(dataset_id)}%",)

    query = """
        with latest_diagnostics as (
            select distinct on (equipment_id)
                diagnostic_report_id,
                equipment_id,
                report_time,
                technical_state,
                degradation_index::double precision as degradation_index,
                anomaly_score::double precision as anomaly_score,
                summary
            from app.diagnostic_reports
            order by equipment_id, report_time desc, diagnostic_report_id desc
        ),
        latest_rul as (
            select distinct on (equipment_id)
                rul_forecast_id,
                equipment_id,
                diagnostic_report_id,
                forecast_time,
                predicted_rul_hours::double precision as predicted_rul_hours,
                lower_bound_hours::double precision as lower_bound_hours,
                upper_bound_hours::double precision as upper_bound_hours,
                confidence_score::double precision as confidence_score,
                model_name,
                model_version
            from app.rul_forecasts
            order by equipment_id, forecast_time desc, rul_forecast_id desc
        ),
        latest_recommendations as (
            select distinct on (equipment_id)
                recommendation_id,
                equipment_id,
                rul_forecast_id,
                diagnostic_report_id,
                recommended_work_id,
                priority,
                recommendation_text,
                recommended_start_date,
                recommended_deadline
            from app.maintenance_recommendations
            order by equipment_id, recommendation_id desc
        )
        select
            e.equipment_id,
            e.equipment_code,
            e.equipment_name,
            e.location,
            e.criticality_level,
            d.diagnostic_report_id,
            d.report_time,
            d.technical_state,
            d.degradation_index,
            d.anomaly_score,
            d.summary,
            r.rul_forecast_id,
            r.forecast_time,
            r.predicted_rul_hours,
            r.lower_bound_hours,
            r.upper_bound_hours,
            r.confidence_score,
            r.model_name,
            r.model_version,
            rec.recommendation_id,
            rec.priority,
            rec.recommendation_text,
            rec.recommended_start_date,
            rec.recommended_deadline,
            mw.work_code as recommended_work_code,
            mw.work_name as recommended_work_name
        from app.equipment e
        left join latest_diagnostics d on d.equipment_id = e.equipment_id
        left join latest_rul r on r.equipment_id = e.equipment_id
        left join latest_recommendations rec on rec.equipment_id = e.equipment_id
        left join app.maintenance_works mw on mw.work_id = rec.recommended_work_id
        where {where_clause}
        order by e.equipment_code
    """.format(where_clause=where_clause)
    with connection.cursor(row_factory=dict_row) as cursor:
        cursor.execute(query, where_params)
        return list(cursor.fetchall())
