# tests/unit/raw/test_opc_contract.py
"""The byte-preservation contract — the foundation of the raw layer.

If these fail, nothing else in contextifier.raw can be trusted.
"""

from __future__ import annotations

import io
import zipfile

import pytest

from contextifier.errors import ContextifierError
from contextifier.raw.opc import OpcPackage
from contextifier.raw.xmlpart import NS, XmlPart, qn


def _mini_package() -> bytes:
    """A minimal OPC-shaped zip with several parts of varied content."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Default Extension="bin" ContentType="application/octet-stream"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/wb+xml"/>'
            "</Types>",
        )
        z.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://x/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>",
        )
        z.writestr("xl/workbook.xml", "<workbook><sheets/></workbook>")
        # Deliberately quirky bytes: BOM, weird whitespace, high compression value
        z.writestr("xl/quirky.xml", b"\xef\xbb\xbf<a  attr='1'\n\t>text</a>")
        z.writestr("media/blob.bin", bytes(range(256)) * 100)
        z.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId7" Type="http://x/chart" Target="charts/chart1.xml"/>'
            "</Relationships>",
        )
        z.writestr("xl/charts/chart1.xml", "<chart/>")
    return buf.getvalue()


class TestBytePreservation:
    def test_untouched_roundtrip_is_content_identical(self):
        src = _mini_package()
        out = OpcPackage.open(src).to_bytes()
        a, b = zipfile.ZipFile(io.BytesIO(src)), zipfile.ZipFile(io.BytesIO(out))
        assert sorted(a.namelist()) == sorted(b.namelist())
        for name in a.namelist():
            assert a.read(name) == b.read(name), f"content drift in {name}"

    def test_single_edit_touches_only_that_part(self):
        src = _mini_package()
        pkg = OpcPackage.open(src)
        pkg.get_part("xl/workbook.xml").write(
            b"<workbook><sheets><s/></sheets></workbook>"
        )
        out = pkg.to_bytes()
        a, b = zipfile.ZipFile(io.BytesIO(src)), zipfile.ZipFile(io.BytesIO(out))
        for name in a.namelist():
            if name == "xl/workbook.xml":
                assert b.read(name) == b"<workbook><sheets><s/></sheets></workbook>"
            else:
                assert a.read(name) == b.read(name), f"collateral change in {name}"

    def test_quirky_bytes_survive_verbatim(self):
        """BOM + odd whitespace must not be normalized for clean parts."""
        src = _mini_package()
        pkg = OpcPackage.open(src)
        pkg.get_part("xl/workbook.xml").write(b"<workbook/>")  # dirty a sibling
        out = zipfile.ZipFile(io.BytesIO(pkg.to_bytes()))
        assert out.read("xl/quirky.xml") == b"\xef\xbb\xbf<a  attr='1'\n\t>text</a>"

    def test_lazy_xmlpart_read_does_not_dirty(self):
        """Parsing (without mutation) must never rewrite a part."""
        src = _mini_package()
        pkg = OpcPackage.open(src)
        xp = XmlPart(pkg.get_part("xl/quirky.xml"))
        assert xp.root.tag == "a"  # parses fine
        xp.flush()  # no-op: not dirty
        out = zipfile.ZipFile(io.BytesIO(pkg.to_bytes()))
        assert out.read("xl/quirky.xml") == b"\xef\xbb\xbf<a  attr='1'\n\t>text</a>"

    def test_add_and_remove_parts(self):
        pkg = OpcPackage.open(_mini_package())
        pkg.add_part("xl/new.xml", b"<new/>")
        pkg.remove_part("media/blob.bin")
        out = zipfile.ZipFile(io.BytesIO(pkg.to_bytes()))
        assert out.read("xl/new.xml") == b"<new/>"
        assert "media/blob.bin" not in out.namelist()


class TestPackageBasics:
    def test_open_from_path_bytes_and_stream(self, tmp_path):
        data = _mini_package()
        p = tmp_path / "x.zip"
        p.write_bytes(data)
        assert OpcPackage.open(p).part_names == OpcPackage.open(data).part_names
        assert OpcPackage.open(io.BytesIO(data)).has_part("xl/workbook.xml")

    def test_bad_magic_rejected(self):
        with pytest.raises(ContextifierError, match="ZIP"):
            OpcPackage.open(b"not a zip at all")

    def test_sniff_format(self):
        assert OpcPackage.open(_mini_package()).sniff_format() == "xlsx"

    def test_content_type_lookup_and_override(self):
        pkg = OpcPackage.open(_mini_package())
        assert pkg.content_type_of("xl/workbook.xml") == "application/wb+xml"
        assert pkg.content_type_of("media/blob.bin") == "application/octet-stream"
        pkg.set_content_type_override("xl/charts/chart1.xml", "application/chart+xml")
        assert pkg.content_type_of("xl/charts/chart1.xml") == "application/chart+xml"


class TestRelationships:
    def test_rels_iteration_and_resolution(self):
        pkg = OpcPackage.open(_mini_package())
        rels = pkg.rels_for("xl/workbook.xml")
        assert rels is not None
        charts = rels.by_type("/chart")
        assert charts[0]["id"] == "rId7"
        assert (
            rels.resolve("xl/workbook.xml", charts[0]["target"])
            == "xl/charts/chart1.xml"
        )
        assert rels.target_of("rId7") == "charts/chart1.xml"

    def test_package_root_rels(self):
        pkg = OpcPackage.open(_mini_package())
        root_rels = pkg.rels_for("")
        assert (
            root_rels is not None and root_rels.target_of("rId1") == "xl/workbook.xml"
        )

    def test_add_remove_and_next_id(self):
        pkg = OpcPackage.open(_mini_package())
        rels = pkg.rels_for("xl/workbook.xml")
        rid = rels.next_id()
        assert rid != "rId7"
        rels.add(rid, "http://x/image", "media/blob.bin")
        assert rels.target_of(rid) == "media/blob.bin"
        assert rels.remove(rid) is True
        assert rels.target_of(rid) is None


class TestQn:
    def test_qn_expands_known_prefixes(self):
        assert qn("w:p") == f"{{{NS['w']}}}p"
        assert qn("c:ser") == f"{{{NS['c']}}}ser"

    def test_qn_rejects_unknown_prefix(self):
        with pytest.raises(KeyError, match="prefix"):
            qn("zz:nope")


class TestRepeatedSave:
    def test_double_to_bytes_is_stable(self):
        """writestr must not mutate source ZipInfo state (regression:
        header_offset corruption made the second save raise BadZipFile)."""
        pkg = OpcPackage.open(_mini_package())
        pkg.get_part("xl/workbook.xml").write(b"<workbook/>")
        first = pkg.to_bytes()
        second = pkg.to_bytes()  # must not raise
        a, b = zipfile.ZipFile(io.BytesIO(first)), zipfile.ZipFile(io.BytesIO(second))
        for name in a.namelist():
            assert a.read(name) == b.read(name)
