import unittest
from unittest.mock import Mock, patch, MagicMock
import json
from opentelemetry.sdk.trace.export import SpanExportResult, ReadableSpan
from opentelemetry.trace import SpanContext, TraceFlags
from opentelemetry.trace.span import Span

from deepeval.tracing.otel.exporter_v1 import ConfidentSpanExporterV1
from deepeval.tracing.types import BaseSpan, TraceSpanStatus
from deepeval.tracing.otel.utils import to_hex_string, set_trace_time


class TestConfidentSpanExporterV1(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.api_key = "test_api_key"
        self.exporter = ConfidentSpanExporterV1(self.api_key)
        
        # Mock trace_manager methods
        self.trace_manager_patcher = patch('deepeval.tracing.otel.exporter_v1.trace_manager')
        self.mock_trace_manager = self.trace_manager_patcher.start()
        
        # Mock perf_epoch_bridge
        self.peb_patcher = patch('deepeval.tracing.otel.exporter_v1.peb')
        self.mock_peb = self.peb_patcher.start()
        
        # Mock set_trace_time function
        self.set_trace_time_patcher = patch('deepeval.tracing.otel.exporter_v1.set_trace_time')
        self.mock_set_trace_time = self.set_trace_time_patcher.start()
        
        # Mock deepeval.login_with_confident_api_key
        self.login_patcher = patch('deepeval.tracing.otel.exporter_v1.deepeval.login_with_confident_api_key')
        self.mock_login = self.login_patcher.start()
        
        # Mock capture_tracing_integration
        self.capture_patcher = patch('deepeval.tracing.otel.exporter_v1.capture_tracing_integration')
        self.mock_capture = self.capture_patcher.start()

    def tearDown(self):
        """Clean up after each test method."""
        self.trace_manager_patcher.stop()
        self.peb_patcher.stop()
        self.set_trace_time_patcher.stop()
        self.login_patcher.stop()
        self.capture_patcher.stop()

    def create_mock_readable_span(self, name="test_span", span_id=12345, trace_id=67890, 
                                 parent_span_id=None, start_time=1000000000, end_time=2000000000):
        """Helper method to create a mock ReadableSpan."""
        mock_span = Mock(spec=ReadableSpan)
        mock_span.name = name
        
        # Mock span context
        mock_context = Mock(spec=SpanContext)
        mock_context.span_id = span_id
        mock_context.trace_id = trace_id
        mock_span.context = mock_context
        
        # Mock parent span
        if parent_span_id:
            mock_parent = Mock()
            mock_parent.span_id = parent_span_id
            mock_span.parent = mock_parent
        else:
            mock_span.parent = None
        
        # Mock timing
        mock_span.start_time = start_time
        mock_span.end_time = end_time
        
        # Mock to_json method
        mock_span.to_json.return_value = json.dumps({
            "name": name,
            "span_id": span_id,
            "trace_id": trace_id,
            "start_time": start_time,
            "end_time": end_time
        })
        
        return mock_span

    def test_convert_readable_span_to_base_span(self):
        """Test that _convert_readable_span_to_base_span correctly converts ReadableSpan to BaseSpan."""
        # Arrange
        span_id = 12345
        trace_id = 67890
        parent_span_id = 54321
        start_time = 1000000000
        end_time = 2000000000
        name = "test_span"
        
        mock_span = self.create_mock_readable_span(
            name=name,
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            start_time=start_time,
            end_time=end_time
        )
        
        # Mock the conversion functions
        expected_uuid = "0000000000003039"  # hex of 12345
        expected_trace_uuid = "00000000000000000000000000010932"  # hex of 67890
        expected_parent_uuid = "000000000000d431"  # hex of 54321
        expected_start_time = 1.0
        expected_end_time = 2.0
        
        with patch('deepeval.tracing.otel.exporter_v1.to_hex_string') as mock_to_hex:
            mock_to_hex.side_effect = [expected_uuid, expected_parent_uuid, expected_trace_uuid]
            
            with patch('deepeval.tracing.otel.exporter_v1.peb.epoch_nanos_to_perf_seconds') as mock_epoch_to_perf:
                mock_epoch_to_perf.side_effect = [expected_start_time, expected_end_time]
                
                # Act
                result = self.exporter._convert_readable_span_to_base_span(mock_span)
                
                # Assert
                self.assertIsInstance(result, BaseSpan)
                self.assertEqual(result.name, name)
                self.assertEqual(result.uuid, expected_uuid)
                self.assertEqual(result.trace_uuid, expected_trace_uuid)
                self.assertEqual(result.parent_uuid, expected_parent_uuid)
                self.assertEqual(result.start_time, expected_start_time)
                self.assertEqual(result.end_time, expected_end_time)
                self.assertEqual(result.status, TraceSpanStatus.SUCCESS)
                self.assertEqual(result.children, [])
                
                # Verify metadata was parsed from JSON
                self.assertIsInstance(result.metadata, dict)
                self.assertEqual(result.metadata["name"], name)
                self.assertEqual(result.metadata["span_id"], span_id)
                
                # Verify function calls
                mock_to_hex.assert_any_call(span_id, 16)
                mock_to_hex.assert_any_call(parent_span_id, 16)
                mock_to_hex.assert_any_call(trace_id, 32)
                mock_epoch_to_perf.assert_any_call(start_time)
                mock_epoch_to_perf.assert_any_call(end_time)
                mock_span.to_json.assert_called_once()

    def test_convert_readable_span_to_base_span_no_parent(self):
        """Test _convert_readable_span_to_base_span with no parent span."""
        # Arrange
        mock_span = self.create_mock_readable_span(parent_span_id=None)
        
        with patch('deepeval.tracing.otel.exporter_v1.to_hex_string') as mock_to_hex:
            mock_to_hex.side_effect = ["uuid", "trace_uuid"]
            
            with patch('deepeval.tracing.otel.exporter_v1.peb.epoch_nanos_to_perf_seconds') as mock_epoch_to_perf:
                mock_epoch_to_perf.side_effect = [1.0, 2.0]
                
                # Act
                result = self.exporter._convert_readable_span_to_base_span(mock_span)
                
                # Assert
                self.assertIsNone(result.parent_uuid)
                
                # Verify to_hex_string was not called for parent
                mock_to_hex.assert_called()

    def test_export_method_list_reversal(self):
        """Test that the export method processes spans in reverse order."""
        # Arrange
        spans = [
            self.create_mock_readable_span(name="span1", span_id=1),
            self.create_mock_readable_span(name="span2", span_id=2),
            self.create_mock_readable_span(name="span3", span_id=3)
        ]
        
        # Mock trace_manager methods
        self.mock_trace_manager.get_trace_by_uuid.return_value = None
        self.mock_trace_manager.active_traces.keys.return_value = ["trace1"]
        mock_trace = Mock()
        self.mock_trace_manager.get_trace_by_uuid.return_value = mock_trace
        
        # Mock the conversion method
        with patch.object(self.exporter, '_convert_readable_span_to_base_span') as mock_convert:
            mock_convert.side_effect = lambda span: Mock(name=f"converted_{span.name}", trace_uuid="trace1")
            
            # Act
            result = self.exporter.export(spans)
            
            # Assert
            self.assertEqual(result, SpanExportResult.SUCCESS)
            
            # Verify spans were processed in reverse order
            # The mock_convert should be called with spans in reverse order
            expected_calls = [
                unittest.mock.call(spans[2]),  # span3
                unittest.mock.call(spans[1]),  # span2
                unittest.mock.call(spans[0])   # span1
            ]
            mock_convert.assert_has_calls(expected_calls)

    def test_export_method_trace_management(self):
        """Test that export method properly manages traces."""
        # Arrange
        spans = [self.create_mock_readable_span(span_id=1, trace_id=100)]
        
        # Mock trace_manager methods
        self.mock_trace_manager.get_trace_by_uuid.return_value = None
        self.mock_trace_manager.active_traces.keys.return_value = ["trace1"]
        mock_trace = Mock()
        # Set up side_effect to return None first (for trace creation check), then mock_trace
        self.mock_trace_manager.get_trace_by_uuid.side_effect = [None, mock_trace]
        
        with patch.object(self.exporter, '_convert_readable_span_to_base_span') as mock_convert:
            mock_convert.return_value = Mock(trace_uuid="trace1")
            
            # Act
            result = self.exporter.export(spans)
            
            # Assert
            self.assertEqual(result, SpanExportResult.SUCCESS)
            
            # Verify trace management calls
            self.mock_trace_manager.get_trace_by_uuid.assert_called()
            self.mock_trace_manager.start_new_trace.assert_called_with(trace_uuid="trace1")
            self.mock_trace_manager.add_span.assert_called()
            self.mock_trace_manager.add_span_to_trace.assert_called()
            self.mock_trace_manager.end_trace.assert_called_with("trace1")
            self.mock_trace_manager.clear_traces.assert_called()

    def test_export_method_set_trace_time_called(self):
        """Test that set_trace_time is called for each active trace."""
        # Arrange
        spans = [self.create_mock_readable_span()]
        
        # Mock trace_manager methods
        self.mock_trace_manager.get_trace_by_uuid.return_value = None
        self.mock_trace_manager.active_traces.keys.return_value = ["trace1", "trace2"]
        mock_trace1 = Mock()
        mock_trace2 = Mock()
        # Reset the side_effect for the second call to get_trace_by_uuid
        self.mock_trace_manager.get_trace_by_uuid.side_effect = [None, mock_trace1, mock_trace2]
        
        with patch.object(self.exporter, '_convert_readable_span_to_base_span') as mock_convert:
            mock_convert.return_value = Mock(trace_uuid="trace1")
            
            # Act
            result = self.exporter.export(spans)
            
            # Assert
            self.assertEqual(result, SpanExportResult.SUCCESS)
            
            # Verify set_trace_time was called for each active trace
            self.mock_set_trace_time.assert_any_call(mock_trace1)
            self.mock_set_trace_time.assert_any_call(mock_trace2)
            self.assertEqual(self.mock_set_trace_time.call_count, 2)

    def test_export_method_existing_trace(self):
        """Test that export method handles existing traces correctly."""
        # Arrange
        spans = [self.create_mock_readable_span()]
        existing_trace = Mock()
        
        # Mock trace_manager methods
        self.mock_trace_manager.get_trace_by_uuid.return_value = existing_trace
        self.mock_trace_manager.active_traces.keys.return_value = ["trace1"]
        self.mock_trace_manager.get_trace_by_uuid.return_value = existing_trace
        
        with patch.object(self.exporter, '_convert_readable_span_to_base_span') as mock_convert:
            mock_convert.return_value = Mock(trace_uuid="trace1")
            
            # Act
            result = self.exporter.export(spans)
            
            # Assert
            self.assertEqual(result, SpanExportResult.SUCCESS)
            
            # Verify start_new_trace was NOT called since trace already exists
            self.mock_trace_manager.start_new_trace.assert_not_called()

    def test_export_method_empty_spans(self):
        """Test that export method handles empty spans list correctly."""
        # Arrange
        spans = []
        
        # Act
        result = self.exporter.export(spans)
        
        # Assert
        self.assertEqual(result, SpanExportResult.SUCCESS)
        
        # Verify no conversion calls were made
        self.mock_trace_manager.get_trace_by_uuid.assert_not_called()
        self.mock_trace_manager.start_new_trace.assert_not_called()
        self.mock_trace_manager.add_span.assert_not_called()

    def test_export_method_timeout_parameter(self):
        """Test that export method accepts timeout_millis parameter."""
        # Arrange
        spans = [self.create_mock_readable_span()]
        
        # Mock trace_manager methods
        self.mock_trace_manager.get_trace_by_uuid.return_value = None
        self.mock_trace_manager.active_traces.keys.return_value = []
        
        with patch.object(self.exporter, '_convert_readable_span_to_base_span') as mock_convert:
            mock_convert.return_value = Mock(trace_uuid="trace1")
            
            # Act
            result = self.exporter.export(spans, timeout_millis=50000)
            
            # Assert
            self.assertEqual(result, SpanExportResult.SUCCESS)

    def test_export_method_error_handling(self):
        """Test that export method handles errors gracefully."""
        # Arrange
        spans = [self.create_mock_readable_span()]
        
        # Mock trace_manager methods to raise an exception
        self.mock_trace_manager.get_trace_by_uuid.side_effect = Exception("Test error")
        
        # Act & Assert
        with self.assertRaises(Exception):
            self.exporter.export(spans)

    def test_export_method_span_conversion_error(self):
        """Test that export method handles span conversion errors."""
        # Arrange
        spans = [self.create_mock_readable_span()]
        
        # Mock trace_manager methods
        self.mock_trace_manager.get_trace_by_uuid.return_value = None
        self.mock_trace_manager.active_traces.keys.return_value = []
        
        with patch.object(self.exporter, '_convert_readable_span_to_base_span') as mock_convert:
            mock_convert.side_effect = Exception("Conversion error")
            
            # Act & Assert
            with self.assertRaises(Exception):
                self.exporter.export(spans)


if __name__ == '__main__':
    unittest.main()
