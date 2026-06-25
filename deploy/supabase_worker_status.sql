-- Worker auto-discovery table.
--
-- The worker (Jetson) upserts its CURRENT LAN ip:port into this single row every ~10s.
-- The dashboard reads it to find the worker automatically — no hostname/mDNS (flaky for
-- Python on Windows) and no hand-typed IP (which changes on a phone hotspot).
--
-- Run this ONCE in the Supabase SQL editor (same place you ran supabase_schema.sql).

create table if not exists worker_status (
  name       text primary key,   -- always 'worker' (single row, upserted)
  ip         text,               -- the worker's current LAN IP (e.g. 172.20.10.2)
  port       integer,            -- the worker's HTTP port (default 8077)
  updated_at timestamp           -- last time the worker advertised itself
);

alter table worker_status enable row level security;

drop policy if exists "worker_status all" on worker_status;
create policy "worker_status all" on worker_status
  for all using (true) with check (true);   -- permissive (same as readings/alerts)
