# User Manual

<div>
  <img src="./usermanual_pics/power_on.png" style="width:100%">
  <div align="center">
    1. Power On Device
  </div>
  <br/> <br/>
  <img src="./usermanual_pics/select_language_and_confirm.png" style="width:100%">
  <div align="center">
     2. Select Language and Confirm   
  </div>
  <br/> <br/>
  <img src="./usermanual_pics/speak_to_translate.png" style="width:100%">
  <div align="center">
    3. Speak to Translate
  </div>
  <br/> <br/>
  <img src="./usermanual_pics/return_to_language_select.png" style="width:100%">
  <div align="center">
     4. Option to Return to Language Selection
   </div>
</div>



# Technical Manual

The following contains the technical documentation for the various
software and hardware parts of the project. The following sections are
present:
1. [Pipeline](#pipeline)
2. [Fine-tuning](#fine-tuning)
3. [Audio Board](#audio-board)
4. [Audio Board Setup](#audio-board-setup)
5. [3D Files](#3d-files)
6. [Hardware Assembly](#hardware-assembly)

## File contents on the KLASS laptop

All of the folders referred to here are stored in the home directory.

|Folder|Contents|
|---|---|
| 3D Files | 3D CAD and print files for hardware components |
| Audio\_board | Printed Circuit Board (PCB) related files |
| BOM.xlsx | Bill of materials |
| CrowPanel_ESP32_7.0 | Files for developing Screen UI, flashed to the CrowPanel ESP32 Screen |
| speech-translation-pipeline | The source code for the AI model pipeline |
| speech-translation-capstone-finetune | The source code for the AI model fine-tuning |
| SquarelineStudioUI | Files for developing Screen UI, layout design in Squareline Studio 1.4.2 |

## Pipeline

###  Installation

To be able to build and run the model pipeline, first a Jetson device
must be set up with Docker, CUDA and Jetpack SDK installed. The only
Jetpack version tested is 6.2, but others should work as well. Once the
device is setup the following needs to be done on the device:

```bash
# Setup docker build system
git clone https://github.com/dusty-nv/Jetson-containers ~/Jetson-containers
~/Jetson-containers/install.sh

# Setup the code repositories

# requires authentication to clone
git clone https://github.com/Lutetium-Vanadium/speech-translation-pipeline \
    ~/speech-translation-pipeline 
# must be cloned in the same directory as speech-translation-pipeline
# (in this case we use the home directory)
git clone https://github.com/Lutetium-Vanadium/WhisperS2T ~/whisper_s2t

# Fix for docker build system failing because it uses the wrong runtime to build the project
echo '{
	"runtimes": {
    	"nvidia": {
        	"args": [],
        	"path": "nvidia-container-runtime"
    	}
	},
	"default-runtime": "nvidia"
}' | sudo tee /etc/docker/daemon.json
```

Furthermore, `./audio.py` will need to be modified to get use the correct
device and format for the speaker and microphone.

Additionally, all the `run_*` scripts uses a hard-coded path to allow
running it from any user (required if running as daemon for the device
deployment). Before using any of those scripts, please modify the path
to match what is expected.

A single directory containing data, models and any files that need to be
persisted is mounted as a volume to the docker container. By default
this is the `~/cache` directory, but can be changed in the various
`run_*` files. The following the expected structure of the volume:
```
├── data                                                | Extracted data
│   ├── conversation
│   └── reduced-fleurs
├── models
│   ├── models--segment-any-text--sat-3l-sm             | Sentence segmentation, should be
                                                        | auto generated when run
│   ├── nllb
│   │   └── models--facebook--nllb-200-distilled-600M   | NLLB tokenizer, should be auto-generated
                                                        | when run
│   ├── nllb-ctranslate                                 | Location for CTranslate2 nllb checkpoint
│   │   ├── config.json
│   │   ├── model.bin
│   │   └── shared_vocabulary.json
│   ├── silero-vad                                      | Location for voice activity detector
│   │   └── model.onnx
│   ├── tiny                                            | Word alignment model, should be auto-generated
│   │   ├── config.json
│   │   ├── model.bin
│   │   ├── tokenizer.json
│   │   └── vocabulary.txt
│   ├── tts                                             | All the TTS models
│   │   ├── mms-tts-eng
│   │   ├── mms-tts-hin
│   │   ├── mms-tts-ind
│   │   ├── mms-tts-tgl
│   │   ├── mms-tts-tha
│   │   ├── mms-tts-vie
│   │   ├── mms-tts-zlm
│   │   └── vits-cmn
│   ├── whisper_turbo                                   | Default location for whisper with
│   └── whisper_turbo_weights                           | TensorRT-LLM
└── results                                             | Test results stored here
    ├── conversation
    ├── fleurs
    └── latency
```

All the non auto-generated models can be downloaded from [here](https://huggingface.co/lutetium-vanadium/s2t-pipeline-models)

If you expect to require converting a fine-tuned Whisper checkpoint to
use `TensorRT`, then it is recommended to apply the first fix mentioned
in the [troubleshooting section](#troubleshooting).

### Building

To build the pipeline, run the following:
```bash
./build.sh
```
If you encounter any issue with the build system, check the
[troubleshooting section](#troubleshooting) for any known issues.

In order to convert Whisper model checkpoints, you can modify the
environment variables defined in `./whisper-conversion/convert_trt.sh`
to what you require and run it. Note that the source directory must
contain two files: a copy of `./whisper-conversion/mel_filters.npz` and
`{model}.pt` (a single checkpoint file) where `model` is the name of
the model.

To convert NLLB-200 checkpoints, run the following:
```bash
ct2-transformers-converter \
  --model path/to/transformers/checkpoint \
  --output_dir path/to/models/nllb-ctranslate/ \
  --quantization float16
```

### Running

There are 3 available bash scripts to run the model docker container. 
- `./run_shell.sh` - runs the pipeline expecting the UART screen to be
  available at `/dev/ttyUSB0`.
- `./run_no_screen.sh` - runs the pipeline without making the UART
  screen available to the pipeline.t
- `./run.sh` - same as `run_shell.sh` without an attached interactive
  TTY (see [running on boot](#run-on-boot)).

These by default just execute a bash shell, however, if you pass arguments
to the run script, the arguments will be interpreted as a command and be
executed instead.

The model pipeline can be ran by running the `main.py` file. It has the
following important arguments.
```
usage: main.py [-h] [--file FILE] [--min-chunk-size MIN_CHUNK_SIZE]
               [--model_dir MODEL_DIR]
               [--backend {faster-whisper,whisper_timestamped,mlx-whisper,openai-api,whisper-s2t}]
               [--vac] [--vac-chunk-size VAC_CHUNK_SIZE]
               [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [--prob PROB]

options:
  -h, --help            show this help message and exit
  --file FILE           Filename of 16kHz mono channel wav, on which live
                        streaming is simulated. If not specified the mic
                        is used.
  --min-chunk-size MIN_CHUNK_SIZE (default 1.0)
                        Minimum audio chunk size in seconds. It waits up to
                        this time to do processing. If the processing takes
                        shorter time, it waits, otherwise it processes the
                        whole segment that was received by this time.
  --model_dir MODEL_DIR (default: /model-cache/models/whisper_turbo)
                        Dir where Whisper model and other files are saved.
  --backend {faster-whisper,whisper-s2t}
                        (default: whisper-s2t)
                        Load only this backend for Whisper processing.
                        whisper-s2t is the TensorRT-LLM based Whisper.
  --vac                 Use VAC = voice activity controller. Recommended.
  --vac-chunk-size VAC_CHUNK_SIZE (default: 0.04)
                        VAC sample size in seconds.
  -l, --log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL} (default: DEBUG)
                        Set the log level
  --prob PROB           (default: 0.4) Set the language probabilty threshold.
```

### Run on boot

If you want to run the pipeline as on boot (required during device
deployment), then the following changes need to be made.

- Apply `./Jetson-fixes/daemon-run.patch` so that the docker image can
  run without expecting an interactive TTY attached.

- Setup the `systemd` service file for the pipeline in `/etc/systemd/system/pipeline.service`:
  ```
  [Unit]
  Description="Run pipeline on boot"

  [Service]
  ExecStart=/path/to/speech-translation-pipeline/run_on_boot.sh

  [Install]
  WantedBy=multi-user.target
  ```
- Finally, enable the service:
  ```bash
  sudo systemctl enable pipeline.service
  ```

### Code Structure

- `./main.py` is the entrypoint which contains the linking code to join
  all the models into a single pipeline, and connect the models to the
  hardware.
- `./asr` contains the code for the transcription model (Whisper).
- `./mt` contains the code for the translation model (NLLB-200).
- `./tts` contains the code for the speech-to-text model (MMT-TTS/VITS).
- `./audio.py` interfaces with the speaker and mic.
- `./ui.py` interfaces with the screen over UART.
- `./common.py` contains some common utilities and constants.
- `./test_*` contains the test cases (see [tests](#tests))

### Tests

There are 3 tests:
- `test_fleurs.py` runs the tests on the [reduced-fleurs dataset](https://huggingface.co/datasets/lutetium-vanadium/s2t-test-datasets/resolve/main/reduced-fleurs.tar.gz?download=true), measure the accuracy metrics.
- `test_conversation.py` runs the tests on the [conversation dataset](https://huggingface.co/datasets/lutetium-vanadium/s2t-test-datasets/resolve/main/conversation.tar.gz?download=true), measure the accuracy metrics.
- `test_latency.py` runs the tests on the [reduced-fleurs dataset](https://huggingface.co/datasets/lutetium-vanadium/s2t-test-datasets/resolve/main/reduced-fleurs.tar.gz?download=true), measuring the various latency metrics.

> Before running the tests, make sure to download and extract the data
> to the appropriate location in the mounted volume.

### Troubleshooting

Due to strict versioning and environment requirements, throughout the
projects multiple build issues arose periodically, sometimes with no
change to the codebase. Some of these required applying patches to the
`Jetson-containers` install on the Jetson. It is recommended you try
building and apply the patches only if you face an issue.

- If you require to convert a Whisper checkpoint`TensorRT-LLM`, then
  copy `./Jetson-fixes/tensorrt_llm-source.tar.gz` to
  `Jetson-containers/packages/llm/tensorrt_optimizer/tensorrt_llm/sources/source.tar.gz`.
  This contains necessary changes to the conversion scripts to support
  `whisper-turbo`.
  > Note this is stored as a [Git LFS](https://git-lfs.com/) file, so if you need you will need
  > to either manually download it or install git lfs and pull it.
- If `TensorRT-LLM` fails to build because it cannot install
  `diffusers`, then apply `./Jetson-fixes/tensorrt-diffusers.patch`.
- If the test phase of `transformers` fails, then apply `./Jetson-fixes/transformers.patch`.
- If Whisper gives the following issue, then apply `./Jetson-fixes/whisper.patch`.
  ```
  TypeError: scaled_dot_product_attention(): argument 'is_causal' must be bool, not Tensor
  ```

## Fine-tuning

Code repository containing code for finetuning and evaluating models for the CrossTalk Secure project.

```
├── preprocessing                                       | Preprocessing notebooks for data
│   ├── fleurs-subsets-upload.ipynb 
│   ├── parse_magichub_datasets.ipynb 
│   └── parse_translation_datasets.ipynb 
├── finetuning 
│   ├── nllb-finetune-lora-balanced-multi-corpora.py
│   │   └── Finetune NLLB using LoRA on balanced multilingual corpora.
│   ├── whisper-finetune-lora-cross-val.py
│   │   └── Finetune Whisper with LoRA using cross-validation monolingual subsets.
│   ├── whisper-finetune-lora-unified.py 
│   │   └── Finetune Whisper with LoRA on a unified multilingual dataset.
│   └── whisper-finetune-full-unified.py
│       └── Full finetune of Whisper on unified multilingual dataset.
├── evaluation
│   ├── nllb-evaluate-example.py
│   │   └── Example script showing how to evaluate using NLLB models on sample data.
│   ├── nllb-evaluate-fleurs-reduced-on-whisper-preds.py
│   │   └── Evaluates NLLB translations against Whisper-generated predictions on a
│   │       reduced FLEURS dataset.
│   ├── nllb-evaluate-fleurs-reduced.py
│   │   └── Evaluates NLLB model outputs directly on the reduced FLEURS dataset.
│   ├── nllb-evaluate-with-comet.py
│   │   └── Uses COMET scoring to evaluate NLLB translation quality.
│   ├── whisper-evaluate-ctranslate2.py
│   │   └── Evaluates Whisper outputs generated with CTranslate2 backend.
│   ├── whisper-evaluate-fleurs-reduced-ctranslate2.py
│   │   └── Evaluates Whisper (CTranslate2) predictions on the reduced FLEURS dataset.
│   └── whisper-evaluate-fleurs-reduced.py
│       └── Evaluates Whisper model outputs on the reduced FLEURS dataset (transformers backend).
├── .gitignore
├── README.md
├── jfk.flac
├── whisper_lib.py
├── merge-and-upload.py
├── convert-to-pt.py
└── requirements.txt
```

### Setup

1. **Clone the repository**  
   ```bash
   git clone https://github.com/keeve101/speech-translation-capstone-finetune.git
   cd speech-translation-capstone-finetune
   ```

2. **Create and activate a virtual environment (optional but recommended)**  
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

### Utilities

- `convert-to-pt.py`  
  Converts raw or intermediate model outputs to `.pt` format for compatibility with evaluation scripts.

- `merge-and-upload.py`  
  Merges prediction files and uploads them to cloud or experiment tracking services.

- `whisper_lib.py`  
  Core utility library used by Whisper fine-tuning and evaluation scripts (e.g., data loading, preprocessing, decoding).

- `jfk.flac`  
  Sample audio file (John F. Kennedy speech) used for testing inference pipelines on the evaluate example scripts.

### Finetuning

See the [`finetuning`](./finetuning) directory for training scripts:
- Scripts use either **LoRA** for lightweight updates or full fine-tuning.
- You can modify corpus paths, model configs, or training settings directly in each script.

### Evaluation

Evaluation scripts are in the [`evaluation`](./evaluation) folder and include:
- BLEU and COMET-based scoring
- Support for Whisper and NLLB models
- CTranslate2 and Transformers inference backends

### Preprocessing

The [`preprocessing`](./preprocessing) notebooks:
- Upload and slice subsets of the FLEURS dataset
- Parse open-source datasets (e.g., MagicHub)
- Normalize and prepare text-to-text and speech-to-text corpora

### Usage

#### Finetune Whisper/NLLB:
The `model_path` parameter in the script will be used to load the model checkpoint. 

To run finetuning on the original Whisper model or to continue training from an existing checkpoint you can use the following command:
```bash
python finetuning/whisper-finetune-lora-unified.py 
```

To run finetuning on the original NLLB-200 model or to continue training from an existing checkpoint you can use the following command:
```bash
python finetuning/nllb-finetune-lora-balanced-multi-corpora.py
```

The `output_dir` parameter in the script will be used to save the model checkpoints. Be sure to change the `output_dir` parameter if you intend on training multiple models.

#### Merging LoRA adapters to model checkpoints:
The `merge-and-upload.py` script can be used to merge LoRA adapters to model checkpoints and upload the merged checkpoints to the Hugging Face Hub:
```bash
python merge-and-upload.py \
  --model openai/whisper-large-v3-turbo \
  --adapter_path output-unified-weighted-random-sampler-full-finetune-v2/\
  --repo_owner keeve101 \
  --convert_to_pt True
```
The merged model will be saved onto the `merged_model_path` parameter in the script, initialized as the model path's base name. The `repo_id` parameter `{repo_owner}/{model_name}` will be the repository path on the Hugging Face Hub.

#### Evaluate Whisper/NLLB on FLEURS subset:
To run evaluation on the original NLLB-200 model or a finetuned version, you can use the following command:
```bash
python evaluation/nllb-evaluate-fleurs-reduced.py \
  --model_name keeve101/nllb-200-distilled-600M-finetune-lora-balanced-multi-corpora-checkpoint-100925
```

To run evaluation on the original Whisper model or a finetuned version, you can use the following command:
```bash
python evaluation/nllb-evaluate-fleurs-reduced.py \
  --model_name keeve101/whisper-large-v3-turbo-full-finetune-unified-checkpoint-2400
```
The corresponding evaluation results will be saved onto the `output_file_path` parameter in the script. By default, the parameter is set to `{model_name}-eval.json`.

To support Whisper models on the CTranslate2 backend, you can use the following command to run evaluation on Whisper models with on the CTranslate2 backend:
```bash
python evaluation/whisper-evaluate-fleurs-reduced-ctranslate2.py \
  --model_path keeve101/whisper-large-v3-turbo-full-finetune-unified-checkpoint-2400 \
  --device cuda \ 
  --compute_type int8_float16
```
The converted model will be saved onto the `output_dir` parameter in the script, with the path `{ctranslate2-models/{model_name}`. Conversion will be done using the `ct2-transformers-converter` API from the [CTranslate2 Python library](https://github.com/OpenNMT/CTranslate2). Conversion will not be done if the `output_dir` directory already exists.

#### Evaluate NLLB on FLEURS subset (using COMET):
After running the initial evaluation scripts, you can further evaluate the sources (in original language), references (in target language) and predictions (in target language) using the COMET scoring script:
```bash
python evaluation/nllb-evaluate-with-comet.py \
  --whisper_modeL_used keeve101/whisper-large-v3-turbo-full-finetune-unified-checkpoint-2400 \
  --nllb_model_used facebook/nllb-200-distilled-600M
```

### Notes
#### Previous Finetuning Run Statistics

| Model Name                                             | Script Used                                    | Batch Size | Grad Accum Steps | GPUs Used      | Notes                                      |
|--------------------------------------------------------|------------------------------------------------|------------|------------------|----------------|--------------------------------------------|
| whisper-large-v3-turbo-full-finetune-unified           | `whisper-finetune-full-unified.py`             | 64         | 1                | 1×L40S (48GB)  | Full finetuning on unified multilingual set |
| whisper-large-v3-turbo-lora-unified                    | `whisper-finetune-lora-unified.py`             | 16         | 2               | 1×V100 (32GB)  | Lightweight finetuning using LoRA          |
| nllb-200-distilled-600M-lora-balanced-multi-corpora    | `nllb-finetune-lora-balanced-multi-corpora.py` | 4         | 2                | 1×V100 (32GB)  | Balanced multilingual corpora              |
- Python version used: 3.11
- CUDA version used: 12.8.1
- The batch size and gradient accumulation steps were chosen both empirically and based on recommedations from Whisper authors at [whisper-fine-tune-event](https://github.com/huggingface/community-events/tree/main/whisper-fine-tuning-event).
- According to Whisper authors, the learning rate for finetuning is set to 40x less the initial learning rate used to train the corresponding Whisper model.
- It may also be good to scale the learning rate according to the effective batch size (effective batch size = batch size * gradient accumulation steps). E.g., if you double the effective batch size, you may consider doubling the learning rate as well.

## Audio Board 

The Printed Circuit Board (PCB) related files are stored in the folder ***Audio\_board.*** 

| File/Folder Name  | Description  |
| :---- | :---- |
| Gerber  | Contains all files required for PCB manufacturing  |
| Capstone\_layout | PCB layout  |
| Capstone\_schematic  | PCB schematic |
| bom  | Contains interactive bill of materials (bom) plugin  |
| noise\_cancellation  | Contains python code for noise cancellation that has not been fully tested and deployed |

The current schematic and layout used for manufacturing underwent 2 real-life modifications prior to deployment on the prototype. 

1) Addition of a Schottky diode between the VOUT of the voltage regulator and VCCI (pin 10\) of the USB codec (PCM2902), which was necessary to prevent potential backfeeding and ensure proper voltage regulation.   
2) Before the PCB was designed, the microphone amplifier circuit was tested using a different ADC codec. During those initial tests, it was observed that the output volume was too high, introducing excessive noise. To address this, a voltage divider was added across the amplifier output to reduce the gain. However, in the final PCB design, a different codec was used, which resulted in significantly lower input sensitivity. As a result, resistor R7 from the voltage divider was removed to restore the gain. If further gain increase is required, the PCB allows for modification by cutting specific traces and leaving the gain pin on the MAX9814 microphone amplifier floating, which sets the gain to its maximum of 60dB. 

### Audio Board Setup

![][image1]

1. Connect the pins to the Jetson 40 Pin Header:

   Pin 1 → Ground 

   Pin 2 → 5V

   Pin 3→ I2s\_FS (pin 35 on Orin Nx/ Nano development kit)

   Pin 4→I2S\_SCLK (pin 12 on Orin Nx/ Nano development  kit)

   Pin 5 → DOUT (pin 40 on Orin Nx/ Nano development  kit) 

   I2s\_FS (pin 35 on Orin Nx/ Nano development  kit)


2. Connect the PCB’s USB-C port to the Jetson’s development kit USB type C or type A port using a cable

   ![][image2]

3. Navigate to the Jetson Expansion Header Tool by running the command `$ sudo /opt/nvidia/Jetson-io/Jetson-io.py`

   ![][image3]

4. Select 'Configure Jetson for compatible hardware,' then assign the I2S function to the appropriate pins.
   
5. Save and reboot

### Audio System Modification Based on existing hardware for Enhanced Functionality

To further develop the audio board based on the current hardware, consider replacing the existing I²S Class D audio amplifier with an alternative amplifier that can be more easily integrated with the existing PCM2902 ADC codec. The PCM2902 offers built-in volume and mute controls via its Human Interface Device (HID) function, hence a rotary encoder could also be added for volume control. 

More importantly, integrating such a USB audio codec could enable audio loopback functionality, which is essential for implementing echo cancellation algorithms—an important feature for improving audio clarity in real-time communication scenarios.​

[image1]: audioimages/image1.png

[image2]: audioimages/image2.png

[image3]: audioimages/image3.png

## 3D Files
| Name | Filename | Description |
| :---- | :---- | :---- |
| FormFactor Tablet V6 Fusion | FormFactor\_Tablet\_V6.f3z | Fusion360 file for main prototype design |
| FormFactor Tablet V6 Step | FormFactor\_Tablet\_V6.step | STEP file for main prototype design |
| Bezel | Bezel.stl | STL file for the bezel 3D print (PETG) |
| BackStock | BackStock.stl | STL file for the back 3D print for stock cooling solution (PETG) |
| BackCustom | BackCustom.stl | STL file for the back 3D print for custom cooling solution (PETG) |
| JetsonHolder | JetsonHolder.stl | STL file for the JetsonHolder 3D print (PLA/PETG) |
| BatteryHolder | BatteryHolder.stl | STL file for the BatteryHolder 3D print (PLA/PETG) |

## Hardware Assembly
### Component Assembly
The following section describes several components which require preparation or modification before proceeding with the final assembly.
#### Switch Wiring
<div align="center"><img src="cameraimages/PXL_20250412_164535662.jpg" width="75%" /></div>
Solder ~20cm of wire to 2 XT30 female connectors, with the ends connected to the switch as shown. Apply heatshrink over exposed metal contacts. Label the XT30 connector with only one terminal SW1, and the XT30 connector with two terminals SW2, as shown.

#### Connector Wiring
<div align="center"><img src="cameraimages/PXL_20250412_161134422.jpg" width="75%" /></div>

Solder the output terminals of the mini buck converter to ~5cm of wire, terminating in an XT30 female connector, labelled ESP.

<div align="center"><img src="cameraimages/PXL_20250412_161228615.jpg" width="75%" /></div>
<div align="center"><img src="cameraimages/PXL_20250412_161252844.jpg" width="75%" /></div>
1. Solder the remaining connections as shown in the two images above. The ground wire is connected to the mini buck converter's input, the DC barrel jack, the fan, the XT30 female connector labelled BAT, and the XT30 male connector labelled SW2.

1. Solder the 12V connection between the male XT30 connector labelled SW2 and the female XT30 connector labelled BAT.

1. Solder together a 12V wire between the mini buck converter's input, the DC barrel jack, the fan, and the XT30 male connector labelled SW1.
These various connections use approximately ~5-10cm of wire.

1. Apply heatshrink over exposed metal contacts.

#### Battery holder/wiring
<div align="center"><img src="cameraimages/PXL_20250412_162017264.jpg" width="75%" /></div>
1. Solder the discharge pads on the BMS to ~15-20cm of wire, connected to an XT30 male connector labelled BAT.

<div align="center"><img src="cameraimages/PXL_20250412_162043061.jpg" width="75%" /></div>

2. Solder the batteries in a 3S configuration (connect positive terminal to negative terminal). Solder the 0V, 4.2V, 8.4V, and 12.6V pads to their respective battery terminals as shown. Use ~10-20cm of wire for the pads on the left (0V, 8.4V, charge positive, charge ground), and ~25-40cm of wire for the pads on the right (4.2V, 12.6V).

2. Apply kapton tape on the exposed terminals and fit into battery holder as shown. Fit the buck converter into the battery holder as shown. Wrap ~1cm thick kapton tape around the batteries through the slot in the battery holder.

<div align="center"><img src="cameraimages/PXL_20250412_162111777.jpg" width="75%" /></div>

4. Solder the charge terminals to ~20cm of wire. Solder the positive wire to the Schottky diode. Screw the positive and ground wires into the respective screw terminals on the buck converter's output terminals.

<div align="center"><img src="cameraimages/PXL_20250412_162152136.jpg" width="75%" /></div>

5. Solder the USB-C trigger output terminals to ~15cm of wire, and screw the wires into the respective screw terminals on the buck converter's input terminals.

#### Screen modification
<div align="center"><img src="cameraimages/PXL_20250412_162317757.jpg" width="75%" /></div>

1. Remove the GPIO_D and I2C headers on the screen.

<div align="center"><img src="cameraimages/PXL_20250412_162333928.jpg" width="75%" /></div>

2. Remove the BAT header on the screen. Solder ~5cm of wire connected to an XT30 male connector labelled ESP.

<div align="center"><img src="cameraimages/PXL_20250412_162416801.jpg" width="75%" /></div>

3. Remove the USB-UART module header pins. Solder ~20cm of wire to the GND, RXD, and TXD pins as shown.

<div align="center"><img src="cameraimages/PXL_20250412_162427825.jpg" width="75%" /></div>

4. Crimp the other wire ends into a 4-pin grove connector as shown. Connect it to the UART header on the screen.

<div align="center"><img src="cameraimages/PXL_20250412_162658622.jpg" width="75%" /></div>
Completed screen modification.


### Final assembly
![](fusionimages/1.png)
![](fusionimages/2.png)

1. Place the screen in the bezel as shown. Screw in a M2 5mm screw into the highlighted hole.

![](fusionimages/3.png)

2. Place the assembled battery holder and Jetson holder on the bezel as shown

3. Screw in M2 12mm screws into the 3 holes highligted in red.

4. Place the BMS as shown, highlighted in blue. Secure with kapton tape.

<div align="center"><img src="cameraimages/PXL_20250412_163133047.jpg" width="75%" /></div>
Secured batteries and BMS

![](fusionimages/4.png)

5. Place the USB-C trigger as shown. Secure with kapton tape.
<div align="center"><img src="cameraimages/PXL_20250412_163323040.jpg" width="75%" /></div>
Secured USB-C trigger

![](fusionimages/5.png)

6. Insert the mini buck converter as shown. Secure with kapton tape.

![](fusionimages/6.png)

7. Insert the buck converter as shown. Secure with kapton tape.

<div align="center"><img src="cameraimages/PXL_20250412_163342346.jpg" width="75%" /></div>
Secured mini buck converter

<div align="center"><img src="cameraimages/PXL_20250412_163354188.jpg" width="75%" /></div>

8. Connect the XT30 connectors labelled ESP

![](fusionimages/7.png)

9. Place the audio PCB in the slot as shown in red. Secure with blu-tack adhesive.
10. Place the speaker in the slot as shown in blue. Screw in M2 5mm screws into the 2 holes the speaker is placed over.

<div align="center"><img src="cameraimages/PXL_20250412_162517362.jpg" width="75%" /></div>

11. Connect the speaker and USB-C cable to the PCB as shown.

![](fusionimages/8.png)

12. Place the assembled Jetson on the Jetson holder as shown.
13. Screw in M2 5mm screws into the 2 holes highlighted in red.
14. Screw in M2 20mm screws into the 2 holes highlighted in blue.

![](fusionimages/9.png)

15. If using custom cooling solution, place the fan as shown. Screw in M2 5mm screws into the 2 holes the fan is placed over.

<div align="center"><img src="cameraimages/PXL_20250412_164011976.jpg" width="75%" /></div>
<div align="center"><img src="cameraimages/PXL_20250412_161405987.jpg" width="75%" /></div>

16. Connect the PCB wires to the 40-pin header on the Jetson as shown.

<div align="center"><img src="cameraimages/PXL_20250412_164044292.MP.jpg" width="75%" /></div>

17. Connect the USB-C cable to the Jetson as shown.

<div align="center"><img src="cameraimages/PXL_20250412_164111772.jpg" width="75%" /></div>

18. Connect the USB-UART module to the Jetson as shown.

<div align="center"><img src="cameraimages/PXL_20250412_164239335.jpg" width="75%" /></div>

19. Connect the DC barrel jack to the Jetson as shown.

<div align="center"><img src="cameraimages/PXL_20250412_164321829.MP.jpg" width="75%" /></div>

20. Connect the XT30 connectors labelled BAT.

<div align="center"><img src="cameraimages/PXL_20250412_164547659.jpg" width="75%" /></div>
<div align="center"><img src="cameraimages/PXL_20250412_164559097.jpg" width="75%" /></div>

21. Place the switch into the back as shown.

<div align="center"><img src="cameraimages/PXL_20250412_164633122.jpg" width="75%" /></div>

22. Connect the XT30 connectors labelled SW1. Connect the XT30 connectors labelled SW2.

![](fusionimages/10.png)
<div align="center"><img src="cameraimages/PXL_20250412_164707786.jpg" width="75%" /></div>

23. Enclose the device with the back

![](fusionimages/11.png)
24. Screw in M2 6mm screws into the 2 highlighted holes.
