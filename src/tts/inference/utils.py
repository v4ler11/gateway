import struct

MEDIA_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "ogg": "audio/ogg",
    "pcm": "application/octet-stream"
}


def build_wav_header(
        sample_rate: int,
        channels: int
) -> bytes:
    bytes_per_sample = 4
    audio_format = 3

    byte_rate = sample_rate * channels * bytes_per_sample
    block_align = channels * bytes_per_sample
    bits_per_sample = 32

    return (
            b'RIFF' +
            b'\xff\xff\xff\xff' +
            b'WAVEfmt ' +
            b'\x10\x00\x00\x00' +
            struct.pack('<H', audio_format) +
            struct.pack('<H', channels) +
            struct.pack('<I', sample_rate) +
            struct.pack('<I', byte_rate) +
            struct.pack('<H', block_align) +
            struct.pack('<H', bits_per_sample) +
            b'data' +
            b'\xff\xff\xff\xff'
    )
