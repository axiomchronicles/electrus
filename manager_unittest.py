import unittest
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

from axiomelectrus.handler.filemanager import AsyncFileHandler, AdvancedFileHandler

class TestAdvancedFileHandler(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)
        self.test_file = self.base_path / "test.txt"
        self.test_content = "Hello Electrus!"
        self.test_bytes = self.test_content.encode()

        # Instantiate handler with mocked managers
        self.handler = AdvancedFileHandler(base_path=self.base_path)
        self.handler.lock_manager = MagicMock()
        self.handler.version_manager = MagicMock()
        self.handler.backup_manager = MagicMock()

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("electrus_v2.handler.filemanager.checksumcalculator.ChecksumCalculator.calculate_checksum")
    def test_secure_write_and_read(self, mock_checksum):
        mock_checksum.return_value = "fakehash"

        # Write file securely
        result = self.handler.secure_write(
            file_path=self.test_file,
            content=self.test_content,
            create_version=True,
            verify_integrity=True
        )

        self.assertEqual(result["checksum"], "fakehash")
        self.assertTrue(Path(result["file_path"]).exists())
        self.assertTrue(result["version_created"])

        # Read file securely
        read_result = self.handler.secure_read(
            file_path=self.test_file,
            verify_integrity=True,
            expected_checksum="fakehash"
        )

        self.assertEqual(read_result["content"], self.test_content)
        self.assertEqual(read_result["checksum"], "fakehash")

    @patch("electrus_v2.handler.filemanager.checksumcalculator.ChecksumCalculator.calculate_checksum")
    def test_secure_write_fails_on_integrity(self, mock_checksum):
        mock_checksum.side_effect = lambda path, algo: "wronghash"

        with self.assertRaises(RuntimeError):
            self.handler.secure_write(
                file_path=self.test_file,
                content=self.test_content,
                verify_integrity=True
            )

    @patch("electrus_v2.handler.filemanager.checksumcalculator.ChecksumCalculator.calculate_checksum")
    @patch("electrus_v2.handler.filemanager.checksumcalculator.ChecksumCalculator.verify_integrity")
    def test_secure_read_fails_on_integrity(self, mock_verify, mock_checksum):
        self.test_file.write_text(self.test_content)
        mock_verify.return_value = False
        mock_checksum.return_value = "fake"

        with self.assertRaises(RuntimeError):
            self.handler.secure_read(
                file_path=self.test_file,
                verify_integrity=True,
                expected_checksum="fake"
            )

    @patch("electrus_v2.handler.filemanager.checksumcalculator.ChecksumCalculator.calculate_checksum_async")
    async def test_async_read_write_copy(self, mock_checksum):
        mock_checksum.return_value = "hash123"

        # Write file
        await AsyncFileHandler.write_file(self.test_file, self.test_content)
        self.assertTrue(self.test_file.exists())

        # Read file
        content = await AsyncFileHandler.read_file(self.test_file)
        self.assertEqual(content, self.test_content)

        # Copy file
        target_file = self.base_path / "copied.txt"
        await AsyncFileHandler.copy_file(self.test_file, target_file)
        self.assertTrue(target_file.exists())

        copied_content = await AsyncFileHandler.read_file(target_file)
        self.assertEqual(copied_content, self.test_content)

    @patch("electrus_v2.handler.filemanager.checksumcalculator.ChecksumCalculator.calculate_checksum_async")
    async def test_process_files_batch(self, mock_checksum):
        mock_checksum.return_value = "xyz"
        await AsyncFileHandler.write_file(self.test_file, self.test_content)

        operations = [
            {"type": "read", "path": self.test_file},
            {"type": "write", "path": self.test_file, "content": "Updated"},
            {"type": "checksum", "path": self.test_file},
            {"type": "invalid", "path": self.test_file},
        ]

        results = await AsyncFileHandler.process_files_batch(operations)

        self.assertEqual(results[0]["status"], "success")
        self.assertEqual(results[1]["operation"], "write")
        self.assertEqual(results[2]["checksum"], "xyz")
        self.assertEqual(results[3]["status"], "error")

    def test_create_backup(self):
        self.handler.backup_manager.create_backup.return_value = {
            "status": "ok",
            "files": ["test.txt"]
        }
        result = self.handler.create_backup([self.test_file])
        self.assertEqual(result["status"], "ok")

    def test_list_versions_and_restore(self):
        fake_versions = ["version1", "version2"]
        self.handler.version_manager.list_versions.return_value = fake_versions

        versions = self.handler.list_file_versions("test.txt")
        self.assertEqual(versions, fake_versions)

        self.handler.restore_file_version("test.txt", 1, self.test_file)
        self.handler.version_manager.restore_version.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
