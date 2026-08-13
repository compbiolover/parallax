"""Operational glue for running the pipeline somewhere other than a workstation.

Nothing in here is imported by the pipeline. It is the other direction: this
package knows about the run — where its database lives, which lexicon it needs —
and moves those files between a container's ephemeral disk and durable storage.
Keeping it out of `ingestion` and `daily` is what lets the same image run with no
cloud account at all: if `PARALLAX_STATE_BUCKET` is unset, none of this executes
and the run reads and writes local paths exactly as it does on a laptop.

The one thing here that is not file copying is the lease, and it exists because
the datastore assumes a single writer more strongly than it looks. There is no
WAL, no `busy_timeout` beyond sqlite3's five-second default, and
``Datastore.__init__`` runs the schema script and a migration pass on *every*
open — so two processes opening the same store while a migration is pending is
the sharpest edge in the codebase. Sequencing the scheduled path is the
scheduler's job; the lease is what stops a manual run from colliding with it.
"""
