# Setup (WSL2 / Ubuntu)

`sounddevice` needs the **PortAudio** system library. Install it once:

```bash
sudo apt-get update
sudo apt-get install -y libportaudio2 portaudio19-dev
```

Then install Python dependencies and run:

```bash
pip install -r requirements.txt
python main.py
```
