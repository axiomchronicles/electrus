import unittest
import tempfile
import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from electrus.handler.filemanager.jsonfilemanager import JsonFileHandler


class TestJsonFileHandler(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)
        self.file_name = "test.json"
        self.data = {"name": "Electrus", "type": "NoSQL"}
        
        # Mock Lock and Version managers
        self.mock_lock = MagicMock()
        self.mock_lock.acquire_lock.return_value.__enter__.return_value = None
        self.mock_lock.acquire_lock.return_value.__exit__.return_value = None
        
        self.mock_version = MagicMock()
        self.checksum = "fakechecksum"

        self.handler = JsonFileHandler(
            base_path=self.base_path,
            lock_manager=self.mock_lock,
            version_manager=self.mock_version,
            checksum_algo="sha256"
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("electrus_v2.handler.filemanager.checksumcalculator.ChecksumCalculator.calculate_checksum")
    @patch("electrus_v2.handler.filemanager.checksumcalculator.ChecksumCalculator.verify_integrity")
    def test_write_and_read_sync(self, mock_verify, mock_calc):
        mock_calc.return_value = self.checksum
        mock_verify.return_value = True

        # First, create the file manually so that create_version is triggered
        path = self.base_path / self.file_name
        with open(path, "w") as f:
            json.dump({"bootstrap": True}, f)

        # Test write
        result = self.handler.write(
            file=self.file_name,
            data=self.data,
            create_version=True,
            verify_integrity=True
        )

        self.assertTrue(Path(result["file_path"]).exists())
        self.assertEqual(result["checksum"], self.checksum)
        self.assertTrue(result["versioned"])
        self.mock_version.create_version.assert_called_once()

        # Test read
        read_result = self.handler.read(
            file=self.file_name,
            verify_integrity=True,
            expected_checksum=self.checksum
        )

        self.assertEqual(read_result["data"], self.data)
        self.assertEqual(read_result["checksum"], self.checksum)


    @patch("electrus_v2.handler.filemanager.ChecksumCalculator.calculate_checksum_async")
    @patch("electrus_v2.handler.filemanager.ChecksumCalculator.calculate_checksum")
    @patch("electrus_v2.handler.filemanager.ChecksumCalculator.verify_integrity")
    async def test_write_and_read_async(self, mock_verify, mock_calc_sync, mock_calc_async):
        mock_calc_sync.return_value = self.checksum
        mock_calc_async.return_value = self.checksum
        mock_verify.return_value = True

        # Write asynchronously
        write_result = await self.handler.write_async(
            file=self.file_name,
            data=self.data,
            create_version=True,
            verify_integrity=True
        )

        self.assertTrue(Path(write_result["file_path"]).exists())
        self.assertEqual(write_result["checksum"], self.checksum)

        # Read asynchronously
        read_result = await self.handler.read_async(
            file=self.file_name,
            verify_integrity=True,
            expected_checksum=self.checksum
        )

        self.assertEqual(read_result["data"], self.data)
        self.assertEqual(read_result["checksum"], self.checksum)


    def test_file_not_found_sync(self):
        with self.assertRaises(FileNotFoundError):
            self.handler.read(file="non_existent.json")

    async def test_file_not_found_async(self):
        with self.assertRaises(FileNotFoundError):
            await self.handler.read_async(file="non_existent.json")


if __name__ == "__main__":
    unittest.main(verbosity=2)
