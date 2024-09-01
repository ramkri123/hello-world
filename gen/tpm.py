#hardware tpm created key attestation example code
from tpm2_pytss import *
import hashlib

def create_primary_key(context):
    """Create a primary key in the TPM."""
    primary_key = context.create_primary(
        TPM2_RH_OWNER,
        TPM2_ALG_RSA,
        TPM2B_PUBLIC(
            TPM2_ALG_RSA,
            2048,
            TPM2B_PUBLIC_RSA
        ),
        TPM2B_SENSITIVE_CREATE(
            TPM2_ALG_RSA,
            TPM2B_AUTH(b"password")
        )
    )
    return primary_key['handle']

def create_key(context, parent_handle):
    """Create a child key under the given parent handle."""
    key = context.create(
        parent_handle,
        TPM2B_SENSITIVE_CREATE(
            TPM2_ALG_RSA,
            TPM2B_AUTH(b"password")
        ),
        TPM2B_PUBLIC(
            TPM2_ALG_RSA,
            2048,
            TPM2B_PUBLIC_RSA
        )
    )
    return key['handle'], key['public']

def get_quote(context, key_handle, primary_key_handle):
    """Get an attestation quote using the primary key."""
    # Prepare data to quote
    data = b"Attestation data"
    digest = hashlib.sha256(data).digest()

    # Generate quote
    quote = context.quote(
        key_handle,
        digest,
        TPM2_ALG_SHA256
    )
    return quote

def main():
    # Initialize TPM2 context
    context = TSS2_SYS_CONTEXT()
    context.init()

    # Create a primary key
    primary_key_handle = create_primary_key(context)

    # Create a secondary key for signing the quote
    key_handle, _ = create_key(context, primary_key_handle)

    # Get an attestation quote
    quote = get_quote(context, key_handle, primary_key_handle)
    print("Quote:", quote)

    # Cleanup
    context.flush_context(key_handle)
    context.flush_context(primary_key_handle)
    context.cleanup()

if __name__ == "__main__":
    main()

