from supabase_client import supabase

BUCKET_NAME = "pdf-documents"


def find_existing_pdf(owner_id, filename):
    result = (
        supabase.table("documents")
        .select("id, filename, storage_path")
        .eq("owner_id", owner_id)
        .eq("filename", filename)
        .limit(1)
        .execute()
    )

    return result.data[0] if result.data else None


def upload_pdf_to_supabase(
    local_path,
    filename,
    session_id,
    page_count=0,
    chunk_count=0,
    content_hash=None,
):
    # session_id temporarily carries the persistent owner_id.
    owner_id = session_id
    storage_path = f"{owner_id}/{filename}"

    existing = find_existing_pdf(owner_id, filename)

    if existing:
        return existing

    with open(local_path, "rb") as pdf_file:
        supabase.storage.from_(BUCKET_NAME).upload(
            path=storage_path,
            file=pdf_file,
            file_options={
                "content-type": "application/pdf",
                "upsert": "false",
            },
        )

    record = {
        "owner_id": owner_id,
        "session_id": owner_id,
        "filename": filename,
        "storage_path": storage_path,
        "page_count": page_count,
        "chunk_count": chunk_count,
        "content_hash": content_hash,
        "status": "processed",
    }

    try:
        result = (
            supabase.table("documents")
            .insert(record)
            .execute()
        )
        return result.data[0] if result.data else record

    except Exception:
        # Avoid leaving an orphaned object if metadata insertion fails.
        try:
            supabase.storage.from_(BUCKET_NAME).remove(
                [storage_path]
            )
        except Exception:
            pass
        raise