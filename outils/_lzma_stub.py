"""Remplaçant minimal de _lzma : le Python 3.12 système a été compilé sans liblzma.
Permet d importer torchvision ; toute compression/décompression .xz lève LZMAError.
Correctif propre : sudo apt install liblzma-dev, puis recompiler Python 3.12."""
CHECK_NONE, CHECK_CRC32, CHECK_CRC64, CHECK_SHA256, CHECK_ID_MAX, CHECK_UNKNOWN = 0, 1, 4, 10, 15, 16
FILTER_LZMA1, FILTER_LZMA2, FILTER_DELTA = 0x4000000000000001, 0x21, 0x03
FILTER_X86, FILTER_POWERPC, FILTER_IA64, FILTER_ARM, FILTER_ARMTHUMB, FILTER_SPARC = 4, 5, 6, 7, 8, 9
FORMAT_AUTO, FORMAT_XZ, FORMAT_ALONE, FORMAT_RAW = 0, 1, 2, 3
MF_HC3, MF_HC4, MF_BT2, MF_BT3, MF_BT4 = 0x03, 0x04, 0x12, 0x13, 0x14
MODE_FAST, MODE_NORMAL = 1, 2
PRESET_DEFAULT, PRESET_EXTREME = 6, 0x80000000
class LZMAError(Exception):
    pass
def _absent(*a, **k):
    raise LZMAError("lzma indisponible (Python compilé sans liblzma)")
class LZMACompressor:
    def __init__(self, *a, **k): _absent()
class LZMADecompressor:
    def __init__(self, *a, **k): _absent()
def is_check_supported(check): return False
_encode_filter_properties = _decode_filter_properties = _absent
