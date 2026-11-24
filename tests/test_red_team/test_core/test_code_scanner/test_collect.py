import subprocess
from pathlib import Path

from deepteam.code_scanner import collect_changed_files, collect_files


def _tree(tmp_path: Path) -> Path:
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "agent.py").write_text("x = eval(resp)\n")
    (tmp_path / "app" / "util.ts").write_text("const a = 1;\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.py").write_text("y = 1\n")
    (tmp_path / "pnpm-lock.yaml").write_text("lock\n")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG\x00\x00bin")
    return tmp_path


class TestCollectFiles:
    def test_collects_source_skips_junk(self, tmp_path):
        d = _tree(tmp_path)
        paths = {c.file_path for c in collect_files(str(d))}
        assert "app/agent.py" in paths
        assert "app/util.ts" in paths
        assert not any("node_modules" in p for p in paths)
        assert "pnpm-lock.yaml" not in paths
        assert "logo.png" not in paths

    def test_language_detection(self, tmp_path):
        d = _tree(tmp_path)
        langs = {c.file_path: c.language for c in collect_files(str(d))}
        assert langs["app/agent.py"] == "python"
        assert langs["app/util.ts"] == "typescript"

    def test_include_exclude(self, tmp_path):
        d = _tree(tmp_path)
        only_ts = {c.file_path for c in collect_files(str(d), include=["*.ts"])}
        assert only_ts == {"app/util.ts"}
        no_py = {c.file_path for c in collect_files(str(d), exclude=["*.py"])}
        assert not any(p.endswith(".py") for p in no_py)

    def test_single_file(self, tmp_path):
        d = _tree(tmp_path)
        chunks = collect_files(str(d / "app" / "agent.py"))
        assert chunks and chunks[0].file_path == "agent.py"

    def test_large_file_split_with_line_offsets(self, tmp_path):
        (tmp_path / "big.py").write_text(
            "\n".join(f"x{i} = 1" for i in range(1, 4001)) + "\n"
        )
        chunks = sorted(
            collect_files(str(tmp_path), max_chunk_chars=2000),
            key=lambda c: c.start_line,
        )
        assert len(chunks) > 1
        assert chunks[0].start_line == 1
        # The next chunk starts right after the first chunk's last line.
        assert chunks[1].start_line == 1 + chunks[0].content.count("\n")


class TestCollectChangedFiles:
    def test_diff_mode_returns_only_changed(self, tmp_path):
        def git(*args):
            subprocess.run(
                ["git", "-C", str(tmp_path), *args],
                check=True,
                capture_output=True,
            )

        git("init")
        git("config", "user.email", "t@t.co")
        git("config", "user.name", "t")
        (tmp_path / "a.py").write_text("a = 1\n")
        (tmp_path / "b.py").write_text("b = 1\n")
        git("add", "-A")
        git("commit", "-m", "base")
        (tmp_path / "b.py").write_text("b = eval(x)\n")
        (tmp_path / "c.py").write_text("c = 1\n")
        git("add", "-A")
        git("commit", "-m", "change")

        changed = {
            c.file_path
            for c in collect_changed_files(
                str(tmp_path), base="HEAD~1", head="HEAD"
            )
        }
        assert changed == {"b.py", "c.py"}
