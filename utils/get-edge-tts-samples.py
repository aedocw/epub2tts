#!/usr/bin/env python3

"""
Basic example of edge_tts usage.
"""

import asyncio
import edge_tts

NAMES = ['en-US-AnaNeural', 'en-US-AndrewNeural', 'en-US-AriaNeural', 'en-US-AvaNeural', 'en-US-BrianNeural', 'en-US-ChristopherNeural', 'en-US-EmmaNeural', 'en-US-EricNeural', 'en-US-GuyNeural', 'en-US-JennyNeural', 'en-US-MichelleNeural', 'en-US-RogerNeural', 'en-US-SteffanNeural', 'de-DE-FlorianMultilingualNeural', 'de-DE-SeraphinaMultilingualNeural', 'en-US-AndrewMultilingualNeural', 'en-US-AvaMultilingualNeural', 'en-US-BrianMultilingualNeural', 'en-US-EmmaMultilingualNeural', 'fr-FR-RemyMultilingualNeural', 'fr-FR-VivienneMultilingualNeural']

TEXT = "This is a longer string that I am sending to text to speech, and using the python module directly."

async def amain() -> None:
    """Main function"""
    for name in NAMES:
        output = f"sample-{name}.wav"
        print(f"Saving {output}")
        communicate = edge_tts.Communicate(TEXT, name)
        await communicate.save(output)


if __name__ == "__main__":
    loop = asyncio.get_event_loop_policy().get_event_loop()
    try:
        loop.run_until_complete(amain())
    finally:
        loop.close()
