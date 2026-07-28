"""Reading and writing EPUB, the only container format the app handles.

Split out of translator.py because it is genuinely standalone: it touches no
database, no model and no request state, so it can be read, tested and changed
without knowing anything about the translation pipeline.
"""

import io
import uuid
import zipfile
import html as html_escape
from datetime import datetime as dt
from typing import List

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub as epub_lib


def is_epub_filename(filename: str) -> bool:
    return filename.lower().endswith('.epub')


def extract_epub_book(filepath: str):
    """Read an EPUB and return (chapters, title, author) in spine (reading) order.

    Each chapter is plain text with paragraphs separated by blank lines — enough
    to feed the existing chunk-based translation pipeline. Inline formatting
    (bold/italic/links) is intentionally not preserved; see build_epub_from_chapters.
    """
    book = epub_lib.read_epub(filepath, options={'ignore_ncx': True})

    chapters = []
    for idref, _linear in book.spine:
        item = book.get_item_with_id(idref)
        if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        # The auto-generated nav/TOC document is also an ITEM_DOCUMENT — skip it,
        # it's a list of links to other chapters, not book content.
        if isinstance(item, epub_lib.EpubNav):
            continue
        soup = BeautifulSoup(item.get_content(), 'html.parser')
        blocks = [
            block.get_text(' ', strip=True)
            for block in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'li', 'blockquote'])
        ]
        blocks = [block for block in blocks if block]
        text = '\n\n'.join(blocks) if blocks else soup.get_text('\n\n', strip=True)
        chapters.append(text)

    title = None
    author = None
    title_meta = book.get_metadata('DC', 'title')
    if title_meta:
        title = title_meta[0][0]
    creator_meta = book.get_metadata('DC', 'creator')
    if creator_meta:
        author = creator_meta[0][0]

    return chapters, title, author


def build_epub_from_chapters(chapters: List[str], title: str, author: str) -> bytes:
    """Assemble a minimal, valid EPUB from translated chapter texts.

    The single writer behind both export paths: /download for a translated
    EPUB's chapters, /export/epub for one block of pasted text. Everything
    interpolated into the XML is escaped here, which is the reason to keep
    one copy of it.
    """
    safe_title = html_escape.escape(title or 'Translation')
    safe_author = html_escape.escape(author or 'Book Translator')
    book_uid = str(uuid.uuid4())
    buffer = io.BytesIO()

    manifest_items = []
    spine_items = []
    nav_points = []
    for i, chapter_text in enumerate(chapters, 1):
        manifest_items.append(
            f'<item id="chapter{i}" href="chapter{i}.xhtml" media-type="application/xhtml+xml"/>'
        )
        spine_items.append(f'<itemref idref="chapter{i}"/>')
        nav_points.append(f'''<navPoint id="chapter{i}" playOrder="{i}">
            <navLabel><text>Chapter {i}</text></navLabel>
            <content src="chapter{i}.xhtml"/>
        </navPoint>''')

    content_opf = f'''<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="BookID">
    <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
        <dc:title>{safe_title}</dc:title>
        <dc:creator>{safe_author}</dc:creator>
        <dc:language>en</dc:language>
        <dc:identifier id="BookID">{book_uid}</dc:identifier>
        <meta property="dcterms:modified">{dt.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}</meta>
    </metadata>
    <manifest>
        {''.join(manifest_items)}
        <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
    </manifest>
    <spine toc="ncx">
        {''.join(spine_items)}
    </spine>
</package>'''

    toc_ncx = f'''<?xml version="1.0" encoding="UTF-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
    <head>
        <meta name="dtb:uid" content="{book_uid}"/>
        <meta name="dtb:depth" content="1"/>
    </head>
    <docTitle><text>{safe_title}</text></docTitle>
    <navMap>
        {''.join(nav_points)}
    </navMap>
</ncx>'''

    container_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
    <rootfiles>
        <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
    </rootfiles>
</container>'''

    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as epub:
        epub.writestr('mimetype', 'application/epub+zip', compress_type=zipfile.ZIP_STORED)
        epub.writestr('META-INF/container.xml', container_xml)
        epub.writestr('OEBPS/content.opf', content_opf)
        epub.writestr('OEBPS/toc.ncx', toc_ncx)
        for i, chapter_text in enumerate(chapters, 1):
            paragraphs = [p.strip() for p in chapter_text.split('\n\n') if p.strip()]
            html_paragraphs = ''.join(
                f'<p>{html_escape.escape(p)}</p>\n' for p in paragraphs
            )
            chapter_xhtml = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
    <title>{safe_title}</title>
    <style>
        body {{ font-family: serif; line-height: 1.6; margin: 2em; }}
        p {{ margin-bottom: 1em; text-indent: 1.5em; }}
        p:first-of-type {{ text-indent: 0; }}
    </style>
</head>
<body>
    {html_paragraphs}
</body>
</html>'''
            epub.writestr(f'OEBPS/chapter{i}.xhtml', chapter_xhtml)

    return buffer.getvalue()
