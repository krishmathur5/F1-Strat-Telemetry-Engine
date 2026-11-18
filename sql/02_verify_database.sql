-- Check existing databases
SELECT
    datname AS database_name
FROM pg_database
ORDER BY datname;