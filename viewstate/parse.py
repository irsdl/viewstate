from .colors import COLORS
from .exceptions import ViewStateException
import struct
from datetime import datetime, timedelta

# Exception for viewstate parsing errors
class ViewStateException(Exception):
    pass

# --- Helper functions ---

def read_7bit_encoded_int(b):
    """Reads a 7-bit encoded integer (used for string lengths, etc.)."""
    n = 0
    shift = 0
    i = 0
    while True:
        if i >= len(b):
            raise ViewStateException("Unexpected end of data while reading 7-bit encoded int")
        tmp = b[i]
        i += 1
        n |= (tmp & 0x7F) << shift
        if not (tmp & 0x80):
            break
        shift += 7
    return n, b[i:]

def read_int16(b):
    """Reads a 2-byte signed integer (little-endian)."""
    if len(b) < 2:
        raise ViewStateException("Not enough bytes for int16")
    val = int.from_bytes(b[:2], byteorder='little', signed=True)
    return val, b[2:]

def read_int32(b):
    """Reads a 4-byte unsigned integer (little-endian)."""
    if len(b) < 4:
        raise ViewStateException("Not enough bytes for int32")
    val = int.from_bytes(b[:4], byteorder='little', signed=False)
    return val, b[4:]

def read_int64(b):
    """Reads an 8-byte unsigned integer (little-endian)."""
    if len(b) < 8:
        raise ViewStateException("Not enough bytes for int64")
    val = int.from_bytes(b[:8], byteorder='little', signed=False)
    return val, b[8:]

def read_double(b):
    """Reads an 8-byte double (little-endian)."""
    if len(b) < 8:
        raise ViewStateException("Not enough bytes for double")
    val = struct.unpack('<d', b[:8])[0]
    return val, b[8:]

def read_float(b):
    """Reads a 4-byte float (little-endian)."""
    if len(b) < 4:
        raise ViewStateException("Not enough bytes for float")
    val = struct.unpack('<f', b[:4])[0]
    return val, b[4:]

def parse_dotnet_datetime(ticks):
    """
    Converts .NET ticks (100-nanosecond intervals since 0001-01-01)
    to a Python datetime.
    """
    # .NET ticks are counted from year 1; there are 10,000 ticks in a millisecond.
    return datetime(1, 1, 1) + timedelta(microseconds=ticks // 10)

# A simple mapping for colors (for tokens 0x09 and 0x0A)
COLORS = {
    0: "Black",
    1: "White",
    2: "Red",
    3: "Green",
    4: "Blue",
    # Extend as needed…
}

# --- Parser context (for type and string references) ---

class ParserContext:
    def __init__(self):
        self.type_table = []     # To store types as they are encountered.
        self.string_table = []   # To store strings for indexed-string tokens.
    def add_type(self, t):
        self.type_table.append(t)
        return len(self.type_table) - 1
    def get_type(self, idx):
        return self.type_table[idx]
    def add_string(self, s):
        self.string_table.append(s)
        return len(self.string_table) - 1
    def get_string(self, idx):
        return self.string_table[idx]

# --- Parser base classes using a metaclass for marker registration ---

class ParserMeta(type):
    """
    Metaclass to register parser subclasses by their marker.
    Each subclass sets a “marker” attribute (or tuple of markers).
    """
    def __init__(cls, name, bases, namespace):
        super(ParserMeta, cls).__init__(name, bases, namespace)
        if not hasattr(cls, "registry"):
            cls.registry = {}
        if hasattr(cls, "marker"):
            marker = getattr(cls, "marker")
            if type(marker) not in (tuple, list):
                marker = [marker]
            for m in marker:
                cls.registry[m] = cls

class Parser(metaclass=ParserMeta):
    """
    Base parser class that delegates parsing based on the first byte (marker).
    The full byte array (including marker) and a context are passed.
    """
    @staticmethod
    def parse(b, ctx):
        if not b:
            raise ViewStateException("No data to parse")
        marker = b[0]
        try:
            parser_cls = Parser.registry[marker]
        except KeyError:
            raise ViewStateException("Unknown marker 0x{:02x}".format(marker))
        return parser_cls.parse(b, ctx)

# --- Parsers for constant values ---

class Noop(Parser):
    marker = 0x01
    @staticmethod
    def parse(b, ctx):
        # Consume marker 0x01; returns None.
        return None, b[1:]

class Const(Parser):
    @classmethod
    def parse(cls, b, ctx):
        return cls.const, b[1:]

class NoneConst(Const):
    marker = 0x64
    const = None

class EmptyConst(Const):
    marker = 0x65
    const = ""

class ZeroConst(Const):
    marker = 0x66
    const = 0

class TrueConst(Const):
    marker = 0x67
    const = True

class FalseConst(Const):
    marker = 0x68
    const = False

# --- Parsers for basic numeric and character types ---

class Integer(Parser):
    """
    Parses a 7-bit encoded integer (token 0x02).
    """
    marker = 0x02
    @staticmethod
    def parse(b, ctx):
        if b[0] != 0x02:
            raise ViewStateException("Invalid marker for Integer")
        # Consume marker
        b = b[1:]
        n = 0
        shift = 0
        i = 0
        while True:
            if i >= len(b):
                raise ViewStateException("Unexpected end of data while parsing integer")
            tmp = b[i]
            i += 1
            n |= (tmp & 0x7F) << shift
            if not (tmp & 0x80):
                break
            shift += 7
        return n, b[i:]

class ByteValue(Parser):
    """
    Parses a single byte (token 0x03).
    """
    marker = 0x03
    @staticmethod
    def parse(b, ctx):
        if b[0] != 0x03:
            raise ViewStateException("Invalid marker for Byte")
        return b[1], b[2:]

class CharValue(Parser):
    """
    Parses a character (token 0x04).
    """
    marker = 0x04
    @staticmethod
    def parse(b, ctx):
        if b[0] != 0x04:
            raise ViewStateException("Invalid marker for Char")
        return chr(b[1]), b[2:]

class StringValue(Parser):
    """
    Parses a string (token 0x05).
    First reads a 7-bit encoded integer for length, then that many bytes (UTF-8).
    """
    marker = 0x05
    @staticmethod
    def parse(b, ctx):
        if b[0] != 0x05:
            raise ViewStateException("Invalid marker for String")
        b = b[1:]
        n, b = read_7bit_encoded_int(b)
        if n < 0:
            raise ViewStateException("Invalid string length")
        if n == 0:
            return "", b
        if len(b) < n:
            raise ViewStateException("Not enough bytes for string")
        s = b[:n].decode('utf-8', errors='replace')
        return s, b[n:]

# --- Parsers for more complex types ---

class DateTimeValue(Parser):
    """
    Parses a DateTime (token 0x06) by reading an 8-byte tick value.
    """
    marker = 0x06
    @staticmethod
    def parse(b, ctx):
        if b[0] != 0x06:
            raise ViewStateException("Invalid marker for DateTime")
        b = b[1:]
        ticks, b = read_int64(b)
        dt = parse_dotnet_datetime(ticks)
        return dt, b

class DoubleValue(Parser):
    """
    Parses a double (token 0x07).
    """
    marker = 0x07
    @staticmethod
    def parse(b, ctx):
        if b[0] != 0x07:
            raise ViewStateException("Invalid marker for Double")
        b = b[1:]
        val, b = read_double(b)
        return val, b

class FloatValue(Parser):
    """
    Parses a float (token 0x08).
    """
    marker = 0x08
    @staticmethod
    def parse(b, ctx):
        if b[0] != 0x08:
            raise ViewStateException("Invalid marker for Float")
        b = b[1:]
        val, b = read_float(b)
        return val, b

class RGBA(Parser):
    """
    Parses an RGB(A) color (token 0x09) as a 32-bit integer.
    """
    marker = 0x09
    @staticmethod
    def parse(b, ctx):
        if b[0] != 0x09:
            raise ViewStateException("Invalid marker for RGBA")
        b = b[1:]
        val, b = read_int32(b)
        a = (val >> 24) & 0xFF
        r = (val >> 16) & 0xFF
        g = (val >> 8) & 0xFF
        blue = val & 0xFF
        return "RGBA({}, {}, {}, {})".format(r, g, blue, a), b

class KnownColor(Parser):
    """
    Parses a known color (token 0x0A) using a 7-bit encoded integer as an index.
    """
    marker = 0x0A
    @staticmethod
    def parse(b, ctx):
        if b[0] != 0x0A:
            raise ViewStateException("Invalid marker for KnownColor")
        b = b[1:]
        color_index, b = read_7bit_encoded_int(b)
        try:
            color = COLORS[color_index]
        except KeyError:
            color = "Unknown"
        return "KnownColor: {}".format(color), b

class EnumValue(Parser):
    """
    Parses an enum (token 0x0B). First a type is read (see TypeValue), then a 7-bit encoded int.
    """
    marker = 0x0B
    @staticmethod
    def parse(b, ctx):
        if b[0] != 0x0B:
            raise ViewStateException("Invalid marker for Enum")
        b = b[1:]
        type_ref, b = TypeValue.parse(b, ctx)
        enum_val, b = read_7bit_encoded_int(b)
        return "Enum({}, {})".format(type_ref, enum_val), b

class ColorEmpty(Parser):
    """
    Parses Color.Empty (token 0x0C).
    """
    marker = 0x0C
    @staticmethod
    def parse(b, ctx):
        if b[0] != 0x0C:
            raise ViewStateException("Invalid marker for Color.Empty")
        return "Color.Empty", b[1:]

class PairValue(Parser):
    """
    Parses a pair of values (token 0x0F).
    """
    marker = 0x0F
    @staticmethod
    def parse(b, ctx):
        if b[0] != 0x0F:
            raise ViewStateException("Invalid marker for Pair")
        b = b[1:]
        first, b = Parser.parse(b, ctx)
        second, b = Parser.parse(b, ctx)
        return (first, second), b

class TripletValue(Parser):
    """
    Parses three consecutive values (token 0x10).
    """
    marker = 0x10
    @staticmethod
    def parse(b, ctx):
        if b[0] != 0x10:
            raise ViewStateException("Invalid marker for Triplet")
        b = b[1:]
        first, b = Parser.parse(b, ctx)
        second, b = Parser.parse(b, ctx)
        third, b = Parser.parse(b, ctx)
        return (first, second, third), b

class TypedArray(Parser):
    """
    Parses an array of objects with an explicit type (token 0x14).
    First a type is read, then a 7-bit encoded length, then that many elements.
    """
    marker = 0x14
    @staticmethod
    def parse(b, ctx):
        if b[0] != 0x14:
            raise ViewStateException("Invalid marker for TypedArray")
        b = b[1:]
        type_ref, b = Parser.parse(b, ctx)
        length, b = read_7bit_encoded_int(b)
        arr = []
        for _ in range(length):
            val, b = Parser.parse(b, ctx)
            arr.append(val)
        return arr, b

class StringArray(Parser):
    """
    Parses an array of strings (token 0x15).
    """
    marker = 0x15
    @staticmethod
    def parse(b, ctx):
        if b[0] != 0x15:
            raise ViewStateException("Invalid marker for StringArray")
        b = b[1:]
        n, b = read_7bit_encoded_int(b)
        arr = []
        for _ in range(n):
            s, b = StringValue.parse(b, ctx)
            arr.append(s)
        return arr, b

class ListValue(Parser):
    """
    Parses a list of values (token 0x16).
    """
    marker = 0x16
    @staticmethod
    def parse(b, ctx):
        if b[0] != 0x16:
            raise ViewStateException("Invalid marker for List")
        b = b[1:]
        n, b = read_7bit_encoded_int(b)
        lst = []
        for _ in range(n):
            val, b = Parser.parse(b, ctx)
            lst.append(val)
        return lst, b

class DictValue(Parser):
    """
    Parses a HybridDictionary (token 0x17) or Hashtable (token 0x18).
    For each key/value pair, two values are parsed.
    """
    marker = (0x17, 0x18)
    @staticmethod
    def parse(b, ctx):
        # Consume the marker (either 0x17 or 0x18)
        marker = b[0]
        b = b[1:]
        n, b = read_7bit_encoded_int(b)
        d = {}
        for _ in range(n):
            key, b = Parser.parse(b, ctx)
            value, b = Parser.parse(b, ctx)
            d[key] = value
        return d, b

class TypeValue(Parser):
    """
    Parses a type reference (token 0x19).
    If the next byte is 0x2B, an index is read from the type table;
    otherwise a string is read and stored.
    """
    marker = 0x19
    @staticmethod
    def parse(b, ctx):
        if b[0] != 0x19:
            raise ViewStateException("Invalid marker for Type")
        b = b[1:]
        if b[0] == 0x2B:
            b = b[1:]
            idx, b = read_7bit_encoded_int(b)
            try:
                type_ref = ctx.get_type(idx)
            except IndexError:
                raise ViewStateException("Invalid type reference index")
            return type_ref, b
        else:
            # Read type name as a string and store it.
            type_name, b = StringValue.parse(b, ctx)
            ctx.add_type(type_name)
            return type_name, b

class UnitValue(Parser):
    """
    Parses a Unit (token 0x1B): a double and a 4-byte integer.
    """
    marker = 0x1B
    @staticmethod
    def parse(b, ctx):
        if b[0] != 0x1B:
            raise ViewStateException("Invalid marker for Unit")
        b = b[1:]
        dbl, b = read_double(b)
        int_val, b = read_int32(b)
        return "Unit({}, {})".format(dbl, int_val), b

class UnitEmpty(Parser):
    """
    Parses Unit.Empty (token 0x1C), returning a default Unit.
    """
    marker = 0x1C
    @staticmethod
    def parse(b, ctx):
        if b[0] != 0x1C:
            raise ViewStateException("Invalid marker for Unit.Empty")
        return "Unit(0, 0)", b[1:]

class IndexedString(Parser):
    """
    Parses an IndexedString.
    For token 0x1F, a byte index is read from the string table.
    For token 0x1E, a new string is read and added to the table.
    """
    marker = (0x1E, 0x1F)
    @staticmethod
    def parse(b, ctx):
        token = b[0]
        b = b[1:]
        if token == 0x1F:
            if not b:
                raise ViewStateException("No data for IndexedString reference")
            idx = b[0]
            b = b[1:]
            try:
                s = ctx.get_string(idx)
            except IndexError:
                raise ViewStateException("Invalid string reference index")
            return s, b
        else:  # token == 0x1E
            s, b = StringValue.parse(b, ctx)
            ctx.add_string(s)
            return s, b

class FormattedString(Parser):
    """
    Parses an object converted to string (token 0x28).
    Reads a type reference and then a string.
    """
    marker = 0x28
    @staticmethod
    def parse(b, ctx):
        if b[0] != 0x28:
            raise ViewStateException("Invalid marker for FormattedString")
        b = b[1:]
        type_ref, b = Parser.parse(b, ctx)
        s, b = StringValue.parse(b, ctx)
        if type_ref is not None:
            return "SerialisedObject({})".format(s), b
        else:
            return None, b

class BinaryFormatted(Parser):
    """
    Parses a serialised object (token 0x32).
    Reads a 7-bit encoded length and then that many bytes.
    """
    marker = 0x32
    @staticmethod
    def parse(b, ctx):
        if b[0] != 0x32:
            raise ViewStateException("Invalid marker for BinaryFormatted")
        b = b[1:]
        n, b = read_7bit_encoded_int(b)
        if n > len(b):
            raise ViewStateException("Not enough data for binary formatted object")
        val = b[:n]
        return "BinaryFormatted({})".format(val), b[n:]

class SparseArray(Parser):
    """
    Parses an array containing many nulls (token 0x3C).
    Reads a type, then the full length, then a count of non-null entries.
    For each non-null entry, an index and a value are read.
    """
    marker = 0x3C
    @staticmethod
    def parse(b, ctx):
        if b[0] != 0x3C:
            raise ViewStateException("Invalid marker for SparseArray")
        b = b[1:]
        type_ref, b = Parser.parse(b, ctx)
        length, b = read_7bit_encoded_int(b)
        num_non_null, b = read_7bit_encoded_int(b)
        arr = [None] * length
        for _ in range(num_non_null):
            idx, b = read_7bit_encoded_int(b)
            if idx < 0 or idx >= length:
                raise ViewStateException("Invalid index in sparse array")
            val, b = Parser.parse(b, ctx)
            arr[idx] = val
        return arr, b

# --- Top-level viewstate parser ---

def parse_viewstate(b):
    """
    Parses a viewstate byte array.
    The stream must begin with header bytes 0xff 0x01.
    After parsing the main value, the length of remaining bytes
    determines whether MAC is enabled.
    """
    if len(b) < 2 or b[0] != 0xff or b[1] != 0x01:
        raise ViewStateException("Not a valid ASP.NET 2.0 LOS stream")
    # Skip header
    b = b[2:]
    ctx = ParserContext()
    value, remain = Parser.parse(b, ctx)
    if len(remain) == 0:
        macEnabled = False
    elif len(remain) in (20, 32):
        macEnabled = True
    else:
        raise ViewStateException("Invalid trailing bytes length: {}".format(len(remain)))
    return {"value": value, "macEnabled": macEnabled, "raw": b}

# --- Example usage ---
if __name__ == "__main__":
    # For demonstration purposes, here is a simple viewstate stream.
    # This example uses header (0xff, 0x01) followed by an EmptyConst token (0x65).
    sample = bytes([0xff, 0x01, 0x65])
    try:
        result = parse_viewstate(sample)
        print("Parsed viewstate:", result)
    except ViewStateException as e:
        print("Error parsing viewstate:", e)
