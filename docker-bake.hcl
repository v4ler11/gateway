group "default" {
  targets = ["gat-core", "gat-stt"]
}

target "gat-core" {
  context = "."
  dockerfile = "Dockerfile"
  tags = ["gat-core:latest"]
  output = ["type=docker,compression=uncompressed"]
}

target "gat-stt" {
  context = "."
  dockerfile = "Dockerfile.stt"
  tags = ["gat-stt:latest"]
  output = ["type=docker,compression=uncompressed"]
}
