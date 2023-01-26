import os
import tempfile
import zipfile
import zlib

byte_sequence = os.urandom(1000)
temp_archive = tempfile.NamedTemporaryFile()

with zipfile.ZipFile(temp_archive.name, "w", zipfile.ZIP_DEFLATED) as archive:
    archive.writestr('data.bin', byte_sequence)

with zipfile.ZipFile(temp_archive.name, mode="r") as archive:
    zinfo = archive.infolist()[0]

file_offset = len(zinfo.FileHeader()) + zinfo.header_offset
with open(temp_archive.name, 'rb') as f:
    f.seek(file_offset)
    compressed = f.read(zinfo.compress_size)

decompressed = zlib.decompressobj(-1 * zlib.MAX_WBITS).decompress(compressed)
print(byte_sequence == decompressed)
temp_archive.close()
