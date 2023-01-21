import indexed_gzip as igzip
import zlib
import gzip
import io

def deflate_to_gzip(deflate_stream):
    # Decompress the deflate stream
    decompressed_data = zlib.decompress(deflate_stream, -zlib.MAX_WBITS)

    # Create a gzip file object
    gzip_file = gzip.GzipFile(fileobj=io.BytesIO(decompressed_data))
    
    # Compress the data in gzip format
    gzip_data = gzip_file.read()
    
    return gzip_data

with open('compressed.deflate', 'rb') as f:
    in_stream = f.read()

deflate_to_gzip(in_stream)

# fobj = igzip.IndexedGzipFile('compressed.bin')
# fobj.build_full_index()
# fobj.export_index('slc.gzidx')
