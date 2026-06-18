-- One-time migration: switch readings.ts / alerts.ts from UTC (timestamptz)
-- to LOCAL Malaysia time (plain timestamp), so the raw Supabase rows match
-- your wall clock. Existing rows are converted (UTC -> MYT), not lost.
--
-- Run once in Supabase: SQL Editor -> New query -> paste -> Run.
-- (Only needed if you created the tables with the OLD schema. Fresh installs
--  from supabase_schema.sql are already correct.)

alter table readings alter column ts drop default;
alter table readings
  alter column ts type timestamp using (ts at time zone 'Asia/Kuala_Lumpur');
alter table readings
  alter column ts set default (now() at time zone 'Asia/Kuala_Lumpur');

alter table alerts alter column ts drop default;
alter table alerts
  alter column ts type timestamp using (ts at time zone 'Asia/Kuala_Lumpur');
alter table alerts
  alter column ts set default (now() at time zone 'Asia/Kuala_Lumpur');
