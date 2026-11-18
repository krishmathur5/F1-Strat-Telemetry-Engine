
-- 1. Overall record counts
SELECT 'circuits' AS table_name, COUNT(*) AS records FROM circuits
UNION ALL
SELECT 'sessions', COUNT(*) FROM sessions
UNION ALL
SELECT 'drivers', COUNT(*) FROM drivers
UNION ALL
SELECT 'stints', COUNT(*) FROM stints
UNION ALL
SELECT 'laps', COUNT(*) FROM laps
UNION ALL
SELECT 'telemetry', COUNT(*) FROM telemetry;


-- 2. Lap records by driver
SELECT
    d.driver_code,
    d.full_name,
    COUNT(l.lap_id) AS total_laps
FROM drivers d
JOIN laps l ON d.driver_id = l.driver_id
GROUP BY d.driver_code, d.full_name
ORDER BY d.driver_code;


-- 3. Tyre stint distribution
SELECT
    compound,
    COUNT(*) AS total_stints
FROM stints
GROUP BY compound
ORDER BY total_stints DESC;


-- 4. Lap time statistics
SELECT
    ROUND(AVG(lap_time_ms), 2) AS average_lap_ms,
    ROUND(MIN(lap_time_ms), 2) AS fastest_lap_ms,
    ROUND(MAX(lap_time_ms), 2) AS slowest_lap_ms
FROM laps
WHERE lap_time_ms IS NOT NULL;

--5. Foreign-key relationship validation
SELECT COUNT(*) AS invalid_lap_relationships
FROM laps l
LEFT JOIN stints s
    ON l.stint_id = s.stint_id
    AND l.session_id = s.session_id
    AND l.driver_id = s.driver_id
WHERE s.stint_id IS NULL;