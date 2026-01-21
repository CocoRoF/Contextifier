# contextifier/core/processor/rtf_helper/rtf_text_cleaner.py
"""
RTF Text Cleaner

Functions for removing RTF control codes and cleaning text.
"""
import re
from typing import List

from contextifier.core.processor.rtf_helper.rtf_constants import (
    SHAPE_PROPERTY_NAMES,
    SKIP_DESTINATIONS,
    IMAGE_DESTINATIONS,
)
from contextifier.core.processor.rtf_helper.rtf_decoder import (
    decode_bytes,
)


def clean_rtf_text(text: str, encoding: str = "cp949") -> str:
    """
    Remove RTF control codes and extract pure text.
    
    Uses token-based parsing to prevent content loss.
    
    Args:
        text: RTF text
        encoding: Encoding for decoding
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Protect image tags (replace with temporary markers)
    image_tags = []
    def save_image_tag(m):
        image_tags.append(m.group())
        return f'\x00IMG{len(image_tags)-1}\x00'
    
    text = re.sub(r'\[image:[^\]]+\]', save_image_tag, text)
    
    # Remove shape properties
    text = re.sub(r'\{\\sp\{\\sn\s*\w+\}\{\\sv\s*[^}]*\}\}', '', text)
    text = re.sub(r'shapeType\d+[a-zA-Z0-9]+(?:posrelh\d+posrelv\d+)?', '', text)
    text = re.sub(r'\\shp(?:inst|txt|left|right|top|bottom|bx\w+|by\w+|wr\d+|fblwtxt\d+|z\d+|lid\d+)\b\d*', '', text)
    
    result = []
    i = 0
    n = len(text)
    
    while i < n:
        ch = text[i]
        
        # Restore image tag markers
        if ch == '\x00' and i + 3 < n and text[i+1:i+4] == 'IMG':
            end_idx = text.find('\x00', i + 4)
            if end_idx != -1:
                try:
                    tag_idx = int(text[i+4:end_idx])
                    result.append(image_tags[tag_idx])
                    i = end_idx + 1
                    continue
                except (ValueError, IndexError):
                    pass
        
        if ch == '\\':
            if i + 1 < n:
                next_ch = text[i + 1]
                
                # Special escapes
                if next_ch == '\\':
                    result.append('\\')
                    i += 2
                    continue
                elif next_ch == '{':
                    result.append('{')
                    i += 2
                    continue
                elif next_ch == '}':
                    result.append('}')
                    i += 2
                    continue
                elif next_ch == '~':
                    result.append('\u00A0')  # non-breaking space
                    i += 2
                    continue
                elif next_ch == '-':
                    result.append('\u00AD')  # soft hyphen
                    i += 2
                    continue
                elif next_ch == '_':
                    result.append('\u2011')  # non-breaking hyphen
                    i += 2
                    continue
                elif next_ch == "'":
                    # Hex escape \'XX
                    if i + 3 < n:
                        try:
                            hex_val = text[i+2:i+4]
                            byte_val = int(hex_val, 16)
                            try:
                                result.append(bytes([byte_val]).decode(encoding))
                            except:
                                try:
                                    result.append(bytes([byte_val]).decode('cp1252'))
                                except:
                                    pass
                            i += 4
                            continue
                        except (ValueError, IndexError):
                            pass
                    i += 1
                    continue
                elif next_ch == '*':
                    # \* destination marker, skip
                    i += 2
                    continue
                elif next_ch.isalpha():
                    # Control word: \word[N][delimiter]
                    j = i + 1
                    while j < n and text[j].isalpha():
                        j += 1
                    
                    control_word = text[i+1:j]
                    
                    # Skip numeric parameter
                    while j < n and (text[j].isdigit() or text[j] == '-'):
                        j += 1
                    
                    # Handle delimiter (space is part of control word)
                    if j < n and text[j] == ' ':
                        j += 1
                    
                    # Special control words
                    if control_word in ('par', 'line'):
                        result.append('\n')
                    elif control_word == 'tab':
                        result.append('\t')
                    elif control_word == 'u':
                        # Unicode: \uN?
                        um = re.match(r'\\u(-?\d+)\??', text[i:])
                        if um:
                            try:
                                code = int(um.group(1))
                                if code < 0:
                                    code += 65536
                                result.append(chr(code))
                            except:
                                pass
                            j = i + um.end()
                    
                    i = j
                    continue
            
            i += 1
        elif ch == '{' or ch == '}':
            i += 1
        elif ch == '\r' or ch == '\n':
            i += 1
        else:
            result.append(ch)
            i += 1
    
    text_result = ''.join(result)
    
    # Remove shape property names
    shape_name_pattern = r'\b(' + '|'.join(SHAPE_PROPERTY_NAMES) + r')\b'
    text_result = re.sub(shape_name_pattern, '', text_result)
    
    # Remove garbage numbers
    text_result = re.sub(r'\s*-\d+\s*', ' ', text_result)
    
    # Remove hex data outside image tags
    text_result = _remove_hex_outside_image_tags(text_result)
    
    # Normalize whitespace
    text_result = re.sub(r'\s+', ' ', text_result)
    
    return text_result.strip()


def _remove_hex_outside_image_tags(text: str) -> str:
    """Remove long hex strings outside image tags."""
    protected_ranges = []
    for m in re.finditer(r'\[image:[^\]]+\]', text):
        protected_ranges.append((m.start(), m.end()))
    
    if not protected_ranges:
        return re.sub(r'(?<![a-zA-Z])[0-9a-fA-F]{32,}(?![a-zA-Z])', '', text)
    
    result = []
    last_end = 0
    for start, end in protected_ranges:
        before = text[last_end:start]
        before = re.sub(r'(?<![a-zA-Z])[0-9a-fA-F]{32,}(?![a-zA-Z])', '', before)
        result.append(before)
        result.append(text[start:end])
        last_end = end
    
    after = text[last_end:]
    after = re.sub(r'(?<![a-zA-Z])[0-9a-fA-F]{32,}(?![a-zA-Z])', '', after)
    result.append(after)
    return ''.join(result)


def remove_destination_groups(content: str) -> str:
    """
    Remove RTF destination groups {\\*\\destination...}.
    
    Removes themedata, colorschememapping, latentstyles, datastore, etc.
    to prevent metadata from being extracted as text.
    
    Args:
        content: RTF content
        
    Returns:
        Content with destination groups removed
    """
    result = []
    i = 0
    n = len(content)
    
    while i < n:
        if content[i:i+3] == '{\\*':
            j = i + 3
            while j < n and content[j] in ' \t\r\n':
                j += 1
            
            if j < n and content[j] == '\\':
                k = j + 1
                while k < n and content[k].isalpha():
                    k += 1
                ctrl_word = content[j+1:k]
                
                if ctrl_word in SKIP_DESTINATIONS:
                    depth = 1
                    i += 1
                    while i < n and depth > 0:
                        if content[i] == '{':
                            depth += 1
                        elif content[i] == '}':
                            depth -= 1
                        i += 1
                    continue
                
                if ctrl_word in IMAGE_DESTINATIONS:
                    depth = 1
                    group_start = i
                    i += 1
                    while i < n and depth > 0:
                        if content[i] == '{':
                            depth += 1
                        elif content[i] == '}':
                            depth -= 1
                        i += 1
                    
                    group_content = content[group_start:i]
                    image_tag_match = re.search(r'\[image:[^\]]+\]', group_content)
                    if image_tag_match:
                        tag = image_tag_match.group()
                        if '/uploads/.' not in tag and 'uploads/.' not in tag:
                            result.append(tag)
                    continue
        
        result.append(content[i])
        i += 1
    
    return ''.join(result)


def remove_shape_groups(content: str) -> str:
    """
    Remove shape groups but preserve text in shptxt.
    
    RTF Shape structure:
    {\\shp{\\*\\shpinst...{\\sp{\\sn xxx}{\\sv yyy}}...{\\shptxt actual_text}}}
    
    Args:
        content: RTF content
        
    Returns:
        Content with shape groups cleaned
    """
    result = []
    i = 0
    
    while i < len(content):
        if content[i:i+5] == '{\\shp' or content[i:i+10] == '{\\*\\shpinst':
            depth = 1
            i += 1
            shptxt_content = []
            in_shptxt = False
            shptxt_depth = 0
            
            while i < len(content) and depth > 0:
                if content[i] == '{':
                    if content[i:i+8] == '{\\shptxt':
                        in_shptxt = True
                        shptxt_depth = depth + 1
                        i += 8
                        continue
                    depth += 1
                elif content[i] == '}':
                    if in_shptxt and depth == shptxt_depth:
                        in_shptxt = False
                    depth -= 1
                elif in_shptxt:
                    shptxt_content.append(content[i])
                i += 1
            
            if shptxt_content:
                result.append(''.join(shptxt_content))
        else:
            result.append(content[i])
            i += 1
    
    return ''.join(result)


def remove_shape_property_groups(content: str) -> str:
    """
    Remove shape property groups {\\sp{\\sn xxx}{\\sv yyy}}.
    
    Args:
        content: RTF content
        
    Returns:
        Content with shape properties removed
    """
    content = re.sub(r'\{\\sp\{\\sn\s*[^}]*\}\{\\sv\s*[^}]*\}\}', '', content)
    content = re.sub(r'\{\\sp\s*[^}]*\}', '', content)
    content = re.sub(r'\{\\sn\s*[^}]*\}', '', content)
    content = re.sub(r'\{\\sv\s*[^}]*\}', '', content)
    return content


def remove_shprslt_blocks(content: str) -> str:
    """
    Remove \\shprslt{...} blocks.
    
    Word saves Shape (drawing/table) in \\shp block and duplicates
    the same content in \\shprslt block for backward compatibility.
    
    Args:
        content: RTF content
        
    Returns:
        Content with \\shprslt blocks removed
    """
    result = []
    i = 0
    pattern = '\\shprslt'
    
    while i < len(content):
        idx = content.find(pattern, i)
        if idx == -1:
            result.append(content[i:])
            break
        
        result.append(content[i:idx])
        
        brace_start = content.find('{', idx)
        if brace_start == -1:
            i = idx + len(pattern)
            continue
        
        depth = 1
        j = brace_start + 1
        while j < len(content) and depth > 0:
            if content[j] == '{':
                depth += 1
            elif content[j] == '}':
                depth -= 1
            j += 1
        
        i = j
    
    return ''.join(result)


__all__ = [
    'clean_rtf_text',
    'remove_destination_groups',
    'remove_shape_groups',
    'remove_shape_property_groups',
    'remove_shprslt_blocks',
]
