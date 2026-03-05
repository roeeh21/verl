# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unified Prometheus metrics server using multiprocess mode.

This module provides a /metrics endpoint that collects metrics from all processes
(Ray, vLLM, verl) using prometheus_client's multiprocess mode.

Setup:
    1. Set PROMETHEUS_MULTIPROC_DIR env var before starting Ray and create the directory

    2. Start the unified metrics server if PROMETHEUS_MULTIPROC_DIR is set.

Architecture:
    - All processes write metrics to the shared PROMETHEUS_MULTIPROC_DIR directory
    - The unified /metrics endpoint uses MultiProcessCollector to read from all processes
"""

import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from prometheus_client import CollectorRegistry, generate_latest, multiprocess

logger = logging.getLogger(__name__)


class UnifiedMetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler that serves metrics from all processes via MultiProcessCollector."""

    def do_GET(self):
        if self.path == "/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()

            try:
                # Create a new registry for this request
                registry = CollectorRegistry()

                # Add multiprocess collector to gather from all processes
                multiprocess.MultiProcessCollector(registry)

                # Generate and write metrics
                output = generate_latest(registry)
                self.wfile.write(output)

            except Exception as e:
                error_msg = f"# Error generating metrics: {e}\n"
                self.wfile.write(error_msg.encode("utf-8"))
                logger.exception("Error generating metrics")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress default logging
        pass


def start_unified_metrics_server(port: int = 9090) -> HTTPServer | None:
    """
    Start the unified metrics server that collects from all processes.

    This server uses MultiProcessCollector to read metrics from all processes
    that write to the shared PROMETHEUS_MULTIPROC_DIR directory passed as an env var.

    Args:
        port: Port to serve metrics on

    Returns:
        HTTPServer instance (runs in background daemon thread), or None if not configured
    """

    server = HTTPServer(("0.0.0.0", port), UnifiedMetricsHandler)
    thread = threading.Thread(name="UnifiedMetricsServer", target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(f"Unified metrics server started on port {port}")
    return server
