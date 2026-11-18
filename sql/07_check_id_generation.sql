
SELECT
    table_name,
    column_name,
    data_type,
    is_identity,
    identity_generation,
    column_default
FROM information_schema.columns
WHERE table_schema = 'public'
AND column_name IN (
    'circuit_id',
    'session_id',
    'driver_id',
    'stint_id',
    'lap_id',
    'telemetry_id'
)
ORDER BY table_name, column_name;
