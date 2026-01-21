The `media_resolution` parameter controls how the Gemini API processes media inputs like images, videos, and PDF documents by determining the **maximum number of tokens** allocated for media inputs, allowing you to balance response quality against latency and cost. For different settings, default values and how they correspond to tokens, see the [Token counts](https://ai.google.dev/gemini-api/docs/media-resolution#token-counts) section.

You can configure media resolution in two ways:

- [Per part](https://ai.google.dev/gemini-api/docs/media-resolution#per-part-media-resolution) (Gemini 3 only)

- [Globally](https://ai.google.dev/gemini-api/docs/media-resolution#global-media-resolution) for an entire `generateContent` request (all multimodal models)

## Per-part media resolution (Gemini 3 only)

Gemini 3 allows you to set media resolution for individual media objects within your request, offering fine-grained optimisation of token usage. You can mix resolution levels in a single request. For example, using high resolution for a complex diagram and low resolution for a simple contextual image. This setting overrides any global configuration for a specific part. For default settings, see [Token counts](https://ai.google.dev/gemini-api/docs/media-resolution#token-counts) section.
**Note:** Per-part media resolution is an experimental feature.  

### Python

    from google import genai
    from google.genai import types

    # The media_resolution parameter for parts is currently only available in the v1alpha API version. (experimental)
    client = genai.Client(
      http_options={
          'api_version': 'v1alpha',
      }
    )

    # Replace with your image data
    with open('path/to/image1.jpg', 'rb') as f:
        image_bytes_1 = f.read()

    # Create parts with different resolutions
    image_part_high = types.Part.from_bytes(
        data=image_bytes_1,
        mime_type='image/jpeg',
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH
    )

    model_name = 'gemini-3-pro-preview'

    response = client.models.generate_content(
        model=model_name,
        contents=["Describe these images:", image_part_high]
    )
    print(response.text)

### Javascript

    // Example: Setting per-part media resolution in JavaScript
    import { GoogleGenAI, MediaResolution, Part } from '@google/genai';
    import * as fs from 'fs';
    import { Buffer } from 'buffer'; // Node.js

    const ai = new GoogleGenAI({ httpOptions: { apiVersion: 'v1alpha' } });

    // Helper function to convert local file to a Part object
    function fileToGenerativePart(path, mimeType, mediaResolution) {
        return {
            inlineData: { data: Buffer.from(fs.readFileSync(path)).toString('base64'), mimeType },
            mediaResolution: { 'level': mediaResolution }
        };
    }

    async function run() {
        // Create parts with different resolutions
        const imagePartHigh = fileToGenerativePart('img.png', 'image/png', Part.MediaResolutionLevel.MEDIA_RESOLUTION_HIGH);
        const model_name = 'gemini-3-pro-preview';
        const response = await ai.models.generateContent({
            model: model_name,
            contents: ['Describe these images:', imagePartHigh]
            // Global config can still be set, but per-part settings will override
            // config: {
            //   mediaResolution: MediaResolution.MEDIA_RESOLUTION_MEDIUM
            // }
        });
        console.log(response.text);
    }
    run();

### REST

    # Replace with paths to your images
    IMAGE_PATH="path/to/image.jpg"

    # Base64 encode the images
    BASE64_IMAGE1=$(base64 -w 0 "$IMAGE_PATH")

    MODEL_ID="gemini-3-pro-preview"

    echo '{
        "contents": [{
          "parts": [
            {"text": "Describe these images:"},
            {
              "inline_data": {
                "mime_type": "image/jpeg",
                "data": "'"$BASE64_IMAGE1"'",
              },
              "media_resolution": {"level": "MEDIA_RESOLUTION_HIGH"}
            }
          ]
        }]
      }' > request.json

    curl -s -X POST \
      "https://generativelanguage.googleapis.com/v1alpha/models/${MODEL_ID}:generateContent" \
      -H "x-goog-api-key: $GEMINI_API_KEY" \
      -H "Content-Type: application/json" \
      -d @request.json

## Global media resolution

You can set a default resolution for all media parts in a request using the
`GenerationConfig`. This is supported by all multimodal models. If a request
includes both global and [per-part settings](https://ai.google.dev/gemini-api/docs/media-resolution#per-part-media-resolution), the per-part setting takes precedence for that specific item.  

### Python

    from google import genai
    from google.genai import types

    client = genai.Client()

    # Prepare standard image part
    with open('image.jpg', 'rb') as f:
        image_bytes = f.read()
    image_part = types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg')

    # Set global configuration
    config = types.GenerateContentConfig(
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH
    )

    response = client.models.generate_content(
        model='gemini-3-flash-preview',
        contents=["Describe this image:", image_part],
        config=config
    )
    print(response.text)

### Javascript

    import { GoogleGenAI, MediaResolution } from '@google/genai';
    import * as fs from 'fs';

    const ai = new GoogleGenAI({ });

    async function run() {
       // ... (Image loading logic) ...

       const response = await ai.models.generateContent({
          model: 'gemini-3-flash-preview',
          contents: ["Describe this image:", imagePart],
          config: {
             mediaResolution: MediaResolution.MEDIA_RESOLUTION_HIGH
          }
       });
       console.log(response.text);
    }
    run();

### REST

    # ... (Base64 encoding logic) ...

    curl -s -X POST \
      "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent" \
      -H "x-goog-api-key: $GEMINI_API_KEY" \
      -H "Content-Type: application/json" \
      -d '{
        "contents": [...],
        "generation_config": {
          "media_resolution": "MEDIA_RESOLUTION_HIGH"
        }
      }'

## Available resolution values

The Gemini API defines the following levels for media resolution:

- `MEDIA_RESOLUTION_UNSPECIFIED`: The default setting. The token count for this level varies significantly between Gemini 3 and earlier Gemini models.
- `MEDIA_RESOLUTION_LOW`: Lower token count, resulting in faster processing and lower cost, but with less detail.
- `MEDIA_RESOLUTION_MEDIUM`: A balance between detail, cost, and latency.
- `MEDIA_RESOLUTION_HIGH`: Higher token count, providing more detail for the model to work with, at the expense of increased latency and cost.
- `MEDIA_RESOLUTION_ULTRA_HIGH` (Per part only): Highest token count, required for specific use cases such as [computer use](https://ai.google.dev/gemini-api/docs/computer-use).

Note that `MEDIA_RESOLUTION_HIGH` provides the optimal performance for most use
cases.

The exact number of tokens generated for each of these
levels depends on both the **media type** (Image, Video, PDF) and the **model
version**.

## Token counts

The tables below summarize the approximate token counts for each
`media_resolution` value and media type per model family.

**Gemini 3 models**

|---|---|---|---|
| **MediaResolution** | **Image** | **Video** | **PDF** |
| `MEDIA_RESOLUTION_UNSPECIFIED` (Default) | 1120 | 70 | 560 |
| `MEDIA_RESOLUTION_LOW` | 280 | 70 | 280 + Native Text |
| `MEDIA_RESOLUTION_MEDIUM` | 560 | 70 | 560 + Native Text |
| `MEDIA_RESOLUTION_HIGH` | 1120 | 280 | 1120 + Native Text |
| `MEDIA_RESOLUTION_ULTRA_HIGH` | 2240 | N/A | N/A |

**Gemini 2.5 models**

|---|---|---|---|---|
| **MediaResolution** | **Image** | **Video** | **PDF (Scanned)** | **PDF (Native)** |
| `MEDIA_RESOLUTION_UNSPECIFIED` (Default) | 256 + Pan \& Scan (\~2048) | 256 | 256 + OCR | 256 + Native Text |
| `MEDIA_RESOLUTION_LOW` | 64 | 64 | 64 + OCR | 64 + Native Text |
| `MEDIA_RESOLUTION_MEDIUM` | 256 | 256 | 256 + OCR | 256 + Native Text |
| `MEDIA_RESOLUTION_HIGH` | 256 + Pan \& Scan | 256 | 256 + OCR | 256 + Native Text |

## Choosing the right resolution

- **Default (`UNSPECIFIED`):** Start with the default. It's tuned for a good balance of quality, latency, and cost for most common use cases.
- **`LOW`:** Use for scenarios where cost and latency are paramount, and fine-grained detail is less critical.
- **`MEDIUM` / `HIGH`:** Increase the resolution when the task requires understanding intricate details within the media. This is often needed for complex visual analysis, chart reading, or dense document comprehension.
- **`ULTRA HIGH`** - Only available for per part setting. Recommended for specific use cases such as computer use or where testing shows a clear enhancement over `HIGH`.
- **Per-part control (Gemini 3):** Optimizes token usage. For example, in a prompt with multiple images, use `HIGH` for a complex diagram and `LOW` or `MEDIUM` for simpler contextual images.

**Recommended settings**

The following lists the recommended media resolution settings for each
supported media type.

|---|---|---|---|
| **Media Type** | **Recommended Setting** | **Max Tokens** | **Usage Guidance** |
| **Images** | `MEDIA_RESOLUTION_HIGH` | 1120 | Recommended for most image analysis tasks to ensure maximum quality. |
| **PDFs** | `MEDIA_RESOLUTION_MEDIUM` | 560 | Optimal for document understanding; quality typically saturates at `medium`. Increasing to `high` rarely improves OCR results for standard documents. |
| **Video** (General) | `MEDIA_RESOLUTION_LOW` (or `MEDIA_RESOLUTION_MEDIUM`) | 70 (per frame) | **Note:** For video, `low` and `medium` settings are treated identically (70 tokens) to optimize context usage. This is sufficient for most action recognition and description tasks. |
| **Video** (Text-heavy) | `MEDIA_RESOLUTION_HIGH` | 280 (per frame) | Required only when the use case involves reading dense text (OCR) or small details within video frames. |

Always test and evaluate the impact of different resolution settings on your
specific application to find the best trade-off between quality, latency, and
cost.

## Version compatibility summary

- The `MediaResolution` enum is available for all models supporting media input.
- The token counts associated with each enum level **differ** between Gemini 3 models and earlier Gemini versions.
- Setting `media_resolution` on individual `Part` objects is **exclusive to
  Gemini 3 models**.

## Next steps

- Learn more about the multimodal capabilities of Gemini API in the [image understanding](https://ai.google.dev/gemini-api/docs/image-understanding), [video understanding](https://ai.google.dev/gemini-api/docs/video-understanding) and [document understanding](https://ai.google.dev/gemini-api/docs/document-processing) guides.

Gemini models are built to be multimodal from the ground up, unlocking a wide range of image processing and computer vision tasks including but not limited to image captioning, classification, and visual question answering without having to train specialized ML models.
| **Tip:** In addition to their general multimodal capabilities, Gemini models (2.0 and newer) offer **improved accuracy** for specific use cases like [object detection](https://ai.google.dev/gemini-api/docs/image-understanding#object-detection) and [segmentation](https://ai.google.dev/gemini-api/docs/image-understanding#segmentation), through additional training. See the [Capabilities](https://ai.google.dev/gemini-api/docs/image-understanding#capabilities) section for more details.

## Passing images to Gemini

You can provide images as input to Gemini using two methods:

- [Passing inline image data](https://ai.google.dev/gemini-api/docs/image-understanding#inline-image): Ideal for smaller files (total request size less than 20MB, including prompts).
- [Uploading images using the File API](https://ai.google.dev/gemini-api/docs/image-understanding#upload-image): Recommended for larger files or for reusing images across multiple requests.

### Passing inline image data

You can pass inline image data in the
request to `generateContent`. You can provide image data as Base64 encoded
strings or by reading local files directly (depending on the language).

The following example shows how to read an image from a local file and pass
it to `generateContent` API for processing.  

### Python

      from google import genai
      from google.genai import types

      with open('path/to/small-sample.jpg', 'rb') as f:
          image_bytes = f.read()

      client = genai.Client()
      response = client.models.generate_content(
        model='gemini-3-flash-preview',
        contents=[
          types.Part.from_bytes(
            data=image_bytes,
            mime_type='image/jpeg',
          ),
          'Caption this image.'
        ]
      )

      print(response.text)

### JavaScript

    import { GoogleGenAI } from "@google/genai";
    import * as fs from "node:fs";

    const ai = new GoogleGenAI({});
    const base64ImageFile = fs.readFileSync("path/to/small-sample.jpg", {
      encoding: "base64",
    });

    const contents = [
      {
        inlineData: {
          mimeType: "image/jpeg",
          data: base64ImageFile,
        },
      },
      { text: "Caption this image." },
    ];

    const response = await ai.models.generateContent({
      model: "gemini-3-flash-preview",
      contents: contents,
    });
    console.log(response.text);

### Go

    bytes, _ := os.ReadFile("path/to/small-sample.jpg")

    parts := []*genai.Part{
      genai.NewPartFromBytes(bytes, "image/jpeg"),
      genai.NewPartFromText("Caption this image."),
    }

    contents := []*genai.Content{
      genai.NewContentFromParts(parts, genai.RoleUser),
    }

    result, _ := client.Models.GenerateContent(
      ctx,
      "gemini-3-flash-preview",
      contents,
      nil,
    )

    fmt.Println(result.Text())

### REST

    IMG_PATH="/path/to/your/image1.jpg"

    if [[ "$(base64 --version 2>&1)" = *"FreeBSD"* ]]; then
    B64FLAGS="--input"
    else
    B64FLAGS="-w0"
    fi

    curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent" \
    -H "x-goog-api-key: $GEMINI_API_KEY" \
    -H 'Content-Type: application/json' \
    -X POST \
    -d '{
        "contents": [{
        "parts":[
            {
                "inline_data": {
                "mime_type":"image/jpeg",
                "data": "'"$(base64 $B64FLAGS $IMG_PATH)"'"
                }
            },
            {"text": "Caption this image."},
        ]
        }]
    }' 2> /dev/null

You can also fetch an image from a URL, convert it to bytes, and pass it to
`generateContent` as shown in the following examples.  

### Python

    from google import genai
    from google.genai import types

    import requests

    image_path = "https://goo.gle/instrument-img"
    image_bytes = requests.get(image_path).content
    image = types.Part.from_bytes(
      data=image_bytes, mime_type="image/jpeg"
    )

    client = genai.Client()

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=["What is this image?", image],
    )

    print(response.text)

### JavaScript

    import { GoogleGenAI } from "@google/genai";

    async function main() {
      const ai = new GoogleGenAI({});

      const imageUrl = "https://goo.gle/instrument-img";

      const response = await fetch(imageUrl);
      const imageArrayBuffer = await response.arrayBuffer();
      const base64ImageData = Buffer.from(imageArrayBuffer).toString('base64');

      const result = await ai.models.generateContent({
        model: "gemini-3-flash-preview",
        contents: [
        {
          inlineData: {
            mimeType: 'image/jpeg',
            data: base64ImageData,
          },
        },
        { text: "Caption this image." }
      ],
      });
      console.log(result.text);
    }

    main();

### Go

    package main

    import (
      "context"
      "fmt"
      "os"
      "io"
      "net/http"
      "google.golang.org/genai"
    )

    func main() {
      ctx := context.Background()
      client, err := genai.NewClient(ctx, nil)
      if err != nil {
          log.Fatal(err)
      }

      // Download the image.
      imageResp, _ := http.Get("https://goo.gle/instrument-img")

      imageBytes, _ := io.ReadAll(imageResp.Body)

      parts := []*genai.Part{
        genai.NewPartFromBytes(imageBytes, "image/jpeg"),
        genai.NewPartFromText("Caption this image."),
      }

      contents := []*genai.Content{
        genai.NewContentFromParts(parts, genai.RoleUser),
      }

      result, _ := client.Models.GenerateContent(
        ctx,
        "gemini-3-flash-preview",
        contents,
        nil,
      )

      fmt.Println(result.Text())
    }

### REST

    IMG_URL="https://goo.gle/instrument-img"

    MIME_TYPE=$(curl -sIL "$IMG_URL" | grep -i '^content-type:' | awk -F ': ' '{print $2}' | sed 's/\r$//' | head -n 1)
    if [[ -z "$MIME_TYPE" || ! "$MIME_TYPE" == image/* ]]; then
      MIME_TYPE="image/jpeg"
    fi

    # Check for macOS
    if [[ "$(uname)" == "Darwin" ]]; then
      IMAGE_B64=$(curl -sL "$IMG_URL" | base64 -b 0)
    elif [[ "$(base64 --version 2>&1)" = *"FreeBSD"* ]]; then
      IMAGE_B64=$(curl -sL "$IMG_URL" | base64)
    else
      IMAGE_B64=$(curl -sL "$IMG_URL" | base64 -w0)
    fi

    curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent" \
        -H "x-goog-api-key: $GEMINI_API_KEY" \
        -H 'Content-Type: application/json' \
        -X POST \
        -d '{
          "contents": [{
            "parts":[
                {
                  "inline_data": {
                    "mime_type":"'"$MIME_TYPE"'",
                    "data": "'"$IMAGE_B64"'"
                  }
                },
                {"text": "Caption this image."}
            ]
          }]
        }' 2> /dev/null

| **Note:** Inline image data limits your total request size (text prompts, system instructions, and inline bytes) to 20MB. For larger requests, [upload image files](https://ai.google.dev/gemini-api/docs/image-understanding#upload-image) using the File API. Files API is also more efficient for scenarios that use the same image repeatedly.

### Uploading images using the File API

For large files or to be able to use the same image file repeatedly, use the
Files API. The following code uploads an image file and then uses the file in a
call to `generateContent`. See the [Files API guide](https://ai.google.dev/gemini-api/docs/files) for
more information and examples.  

### Python

    from google import genai

    client = genai.Client()

    my_file = client.files.upload(file="path/to/sample.jpg")

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[my_file, "Caption this image."],
    )

    print(response.text)

### JavaScript

    import {
      GoogleGenAI,
      createUserContent,
      createPartFromUri,
    } from "@google/genai";

    const ai = new GoogleGenAI({});

    async function main() {
      const myfile = await ai.files.upload({
        file: "path/to/sample.jpg",
        config: { mimeType: "image/jpeg" },
      });

      const response = await ai.models.generateContent({
        model: "gemini-3-flash-preview",
        contents: createUserContent([
          createPartFromUri(myfile.uri, myfile.mimeType),
          "Caption this image.",
        ]),
      });
      console.log(response.text);
    }

    await main();

### Go

    package main

    import (
      "context"
      "fmt"
      "os"
      "google.golang.org/genai"
    )

    func main() {
      ctx := context.Background()
      client, err := genai.NewClient(ctx, nil)
      if err != nil {
          log.Fatal(err)
      }

      uploadedFile, _ := client.Files.UploadFromPath(ctx, "path/to/sample.jpg", nil)

      parts := []*genai.Part{
          genai.NewPartFromText("Caption this image."),
          genai.NewPartFromURI(uploadedFile.URI, uploadedFile.MIMEType),
      }

      contents := []*genai.Content{
          genai.NewContentFromParts(parts, genai.RoleUser),
      }

      result, _ := client.Models.GenerateContent(
          ctx,
          "gemini-3-flash-preview",
          contents,
          nil,
      )

      fmt.Println(result.Text())
    }

### REST

    IMAGE_PATH="path/to/sample.jpg"
    MIME_TYPE=$(file -b --mime-type "${IMAGE_PATH}")
    NUM_BYTES=$(wc -c < "${IMAGE_PATH}")
    DISPLAY_NAME=IMAGE

    tmp_header_file=upload-header.tmp

    # Initial resumable request defining metadata.
    # The upload url is in the response headers dump them to a file.
    curl "https://generativelanguage.googleapis.com/upload/v1beta/files" \
      -H "x-goog-api-key: $GEMINI_API_KEY" \
      -D upload-header.tmp \
      -H "X-Goog-Upload-Protocol: resumable" \
      -H "X-Goog-Upload-Command: start" \
      -H "X-Goog-Upload-Header-Content-Length: ${NUM_BYTES}" \
      -H "X-Goog-Upload-Header-Content-Type: ${MIME_TYPE}" \
      -H "Content-Type: application/json" \
      -d "{'file': {'display_name': '${DISPLAY_NAME}'}}" 2> /dev/null

    upload_url=$(grep -i "x-goog-upload-url: " "${tmp_header_file}" | cut -d" " -f2 | tr -d "\r")
    rm "${tmp_header_file}"

    # Upload the actual bytes.
    curl "${upload_url}" \
      -H "x-goog-api-key: $GEMINI_API_KEY" \
      -H "Content-Length: ${NUM_BYTES}" \
      -H "X-Goog-Upload-Offset: 0" \
      -H "X-Goog-Upload-Command: upload, finalize" \
      --data-binary "@${IMAGE_PATH}" 2> /dev/null > file_info.json

    file_uri=$(jq -r ".file.uri" file_info.json)
    echo file_uri=$file_uri

    # Now generate content using that file
    curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent" \
        -H "x-goog-api-key: $GEMINI_API_KEY" \
        -H 'Content-Type: application/json' \
        -X POST \
        -d '{
          "contents": [{
            "parts":[
              {"file_data":{"mime_type": "'"${MIME_TYPE}"'", "file_uri": "'"${file_uri}"'"}},
              {"text": "Caption this image."}]
            }]
          }' 2> /dev/null > response.json

    cat response.json
    echo

    jq ".candidates[].content.parts[].text" response.json

## Prompting with multiple images

You can provide multiple images in a single prompt by including multiple image
`Part` objects in the `contents` array. These can be a mix of inline data
(local files or URLs) and File API references.  

### Python

    from google import genai
    from google.genai import types

    client = genai.Client()

    # Upload the first image
    image1_path = "path/to/image1.jpg"
    uploaded_file = client.files.upload(file=image1_path)

    # Prepare the second image as inline data
    image2_path = "path/to/image2.png"
    with open(image2_path, 'rb') as f:
        img2_bytes = f.read()

    # Create the prompt with text and multiple images
    response = client.models.generate_content(

        model="gemini-3-flash-preview",
        contents=[
            "What is different between these two images?",
            uploaded_file,  # Use the uploaded file reference
            types.Part.from_bytes(
                data=img2_bytes,
                mime_type='image/png'
            )
        ]
    )

    print(response.text)

### JavaScript

    import {
      GoogleGenAI,
      createUserContent,
      createPartFromUri,
    } from "@google/genai";
    import * as fs from "node:fs";

    const ai = new GoogleGenAI({});

    async function main() {
      // Upload the first image
      const image1_path = "path/to/image1.jpg";
      const uploadedFile = await ai.files.upload({
        file: image1_path,
        config: { mimeType: "image/jpeg" },
      });

      // Prepare the second image as inline data
      const image2_path = "path/to/image2.png";
      const base64Image2File = fs.readFileSync(image2_path, {
        encoding: "base64",
      });

      // Create the prompt with text and multiple images

      const response = await ai.models.generateContent({

        model: "gemini-3-flash-preview",
        contents: createUserContent([
          "What is different between these two images?",
          createPartFromUri(uploadedFile.uri, uploadedFile.mimeType),
          {
            inlineData: {
              mimeType: "image/png",
              data: base64Image2File,
            },
          },
        ]),
      });
      console.log(response.text);
    }

    await main();

### Go

    // Upload the first image
    image1Path := "path/to/image1.jpg"
    uploadedFile, _ := client.Files.UploadFromPath(ctx, image1Path, nil)

    // Prepare the second image as inline data
    image2Path := "path/to/image2.jpeg"
    imgBytes, _ := os.ReadFile(image2Path)

    parts := []*genai.Part{
      genai.NewPartFromText("What is different between these two images?"),
      genai.NewPartFromBytes(imgBytes, "image/jpeg"),
      genai.NewPartFromURI(uploadedFile.URI, uploadedFile.MIMEType),
    }

    contents := []*genai.Content{
      genai.NewContentFromParts(parts, genai.RoleUser),
    }

    result, _ := client.Models.GenerateContent(
      ctx,
      "gemini-3-flash-preview",
      contents,
      nil,
    )

    fmt.Println(result.Text())

### REST

    # Upload the first image
    IMAGE1_PATH="path/to/image1.jpg"
    MIME1_TYPE=$(file -b --mime-type "${IMAGE1_PATH}")
    NUM1_BYTES=$(wc -c < "${IMAGE1_PATH}")
    DISPLAY_NAME1=IMAGE1

    tmp_header_file1=upload-header1.tmp

    curl "https://generativelanguage.googleapis.com/upload/v1beta/files" \
      -H "x-goog-api-key: $GEMINI_API_KEY" \
      -D upload-header1.tmp \
      -H "X-Goog-Upload-Protocol: resumable" \
      -H "X-Goog-Upload-Command: start" \
      -H "X-Goog-Upload-Header-Content-Length: ${NUM1_BYTES}" \
      -H "X-Goog-Upload-Header-Content-Type: ${MIME1_TYPE}" \
      -H "Content-Type: application/json" \
      -d "{'file': {'display_name': '${DISPLAY_NAME1}'}}" 2> /dev/null

    upload_url1=$(grep -i "x-goog-upload-url: " "${tmp_header_file1}" | cut -d" " -f2 | tr -d "\r")
    rm "${tmp_header_file1}"

    curl "${upload_url1}" \
      -H "Content-Length: ${NUM1_BYTES}" \
      -H "X-Goog-Upload-Offset: 0" \
      -H "X-Goog-Upload-Command: upload, finalize" \
      --data-binary "@${IMAGE1_PATH}" 2> /dev/null > file_info1.json

    file1_uri=$(jq ".file.uri" file_info1.json)
    echo file1_uri=$file1_uri

    # Prepare the second image (inline)
    IMAGE2_PATH="path/to/image2.png"
    MIME2_TYPE=$(file -b --mime-type "${IMAGE2_PATH}")

    if [[ "$(base64 --version 2>&1)" = *"FreeBSD"* ]]; then
      B64FLAGS="--input"
    else
      B64FLAGS="-w0"
    fi
    IMAGE2_BASE64=$(base64 $B64FLAGS $IMAGE2_PATH)

    # Now generate content using both images
    curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent" \
        -H "x-goog-api-key: $GEMINI_API_KEY" \
        -H 'Content-Type: application/json' \
        -X POST \
        -d '{
          "contents": [{
            "parts":[
              {"text": "What is different between these two images?"},
              {"file_data":{"mime_type": "'"${MIME1_TYPE}"'", "file_uri": '$file1_uri'}},
              {
                "inline_data": {
                  "mime_type":"'"${MIME2_TYPE}"'",
                  "data": "'"$IMAGE2_BASE64"'"
                }
              }
            ]
          }]
        }' 2> /dev/null > response.json

    cat response.json
    echo

    jq ".candidates[].content.parts[].text" response.json

## Object detection

From Gemini 2.0 onwards, models are further trained to detect objects in an
image and get their bounding box coordinates. The coordinates, relative to image
dimensions, scale to \[0, 1000\]. You need to descale these coordinates based on
your original image size.  

### Python

    from google import genai
    from google.genai import types
    from PIL import Image
    import json

    client = genai.Client()
    prompt = "Detect the all of the prominent items in the image. The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000."

    image = Image.open("/path/to/image.png")

    config = types.GenerateContentConfig(
      response_mime_type="application/json"
      )

    response = client.models.generate_content(model="gemini-3-flash-preview",
                                              contents=[image, prompt],
                                              config=config
                                              )

    width, height = image.size
    bounding_boxes = json.loads(response.text)

    converted_bounding_boxes = []
    for bounding_box in bounding_boxes:
        abs_y1 = int(bounding_box["box_2d"][0]/1000 * height)
        abs_x1 = int(bounding_box["box_2d"][1]/1000 * width)
        abs_y2 = int(bounding_box["box_2d"][2]/1000 * height)
        abs_x2 = int(bounding_box["box_2d"][3]/1000 * width)
        converted_bounding_boxes.append([abs_x1, abs_y1, abs_x2, abs_y2])

    print("Image size: ", width, height)
    print("Bounding boxes:", converted_bounding_boxes)

| **Note:** The model also supports generating bounding boxes based on custom instructions, such as: "Show bounding boxes of all green objects in this image". It also support custom labels like "label the items with the allergens they can contain".

For more examples, check following notebooks in the [Gemini Cookbook](https://github.com/google-gemini/cookbook):

- [2D spatial understanding notebook](https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Spatial_understanding.ipynb)
- [Experimental 3D pointing notebook](https://colab.research.google.com/github/google-gemini/cookbook/blob/main/examples/Spatial_understanding_3d.ipynb)

## Segmentation

Starting with Gemini 2.5, models not only detect items but also segment them
and provide their contour masks.

The model predicts a JSON list, where each item represents a segmentation mask.
Each item has a bounding box ("`box_2d`") in the format `[y0, x0, y1, x1]` with
normalized coordinates between 0 and 1000, a label ("`label`") that identifies
the object, and finally the segmentation mask inside the bounding box, as base64
encoded png that is a probability map with values between 0 and 255.
The mask needs to be resized to match the bounding box dimensions, then
binarized at your confidence threshold (127 for the midpoint).
**Note:** For better results, disable [thinking](https://ai.google.dev/gemini-api/docs/thinking) by setting the thinking budget to 0. See code sample below for an example.  

### Python

    from google import genai
    from google.genai import types
    from PIL import Image, ImageDraw
    import io
    import base64
    import json
    import numpy as np
    import os

    client = genai.Client()

    def parse_json(json_output: str):
      # Parsing out the markdown fencing
      lines = json_output.splitlines()
      for i, line in enumerate(lines):
        if line == "```json":
          json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
          output = json_output.split("```")[0]  # Remove everything after the closing "```"
          break  # Exit the loop once "```json" is found
      return json_output

    def extract_segmentation_masks(image_path: str, output_dir: str = "segmentation_outputs"):
      # Load and resize image
      im = Image.open(image_path)
      im.thumbnail([1024, 1024], Image.Resampling.LANCZOS)

      prompt = """
      Give the segmentation masks for the wooden and glass items.
      Output a JSON list of segmentation masks where each entry contains the 2D
      bounding box in the key "box_2d", the segmentation mask in key "mask", and
      the text label in the key "label". Use descriptive labels.
      """

      config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0) # set thinking_budget to 0 for better results in object detection
      )

      response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[prompt, im], # Pillow images can be directly passed as inputs (which will be converted by the SDK)
        config=config
      )

      # Parse JSON response
      items = json.loads(parse_json(response.text))

      # Create output directory
      os.makedirs(output_dir, exist_ok=True)

      # Process each mask
      for i, item in enumerate(items):
          # Get bounding box coordinates
          box = item["box_2d"]
          y0 = int(box[0] / 1000 * im.size[1])
          x0 = int(box[1] / 1000 * im.size[0])
          y1 = int(box[2] / 1000 * im.size[1])
          x1 = int(box[3] / 1000 * im.size[0])

          # Skip invalid boxes
          if y0 >= y1 or x0 >= x1:
              continue

          # Process mask
          png_str = item["mask"]
          if not png_str.startswith("data:image/png;base64,"):
              continue

          # Remove prefix
          png_str = png_str.removeprefix("data:image/png;base64,")
          mask_data = base64.b64decode(png_str)
          mask = Image.open(io.BytesIO(mask_data))

          # Resize mask to match bounding box
          mask = mask.resize((x1 - x0, y1 - y0), Image.Resampling.BILINEAR)

          # Convert mask to numpy array for processing
          mask_array = np.array(mask)

          # Create overlay for this mask
          overlay = Image.new('RGBA', im.size, (0, 0, 0, 0))
          overlay_draw = ImageDraw.Draw(overlay)

          # Create overlay for the mask
          color = (255, 255, 255, 200)
          for y in range(y0, y1):
              for x in range(x0, x1):
                  if mask_array[y - y0, x - x0] > 128:  # Threshold for mask
                      overlay_draw.point((x, y), fill=color)

          # Save individual mask and its overlay
          mask_filename = f"{item['label']}_{i}_mask.png"
          overlay_filename = f"{item['label']}_{i}_overlay.png"

          mask.save(os.path.join(output_dir, mask_filename))

          # Create and save overlay
          composite = Image.alpha_composite(im.convert('RGBA'), overlay)
          composite.save(os.path.join(output_dir, overlay_filename))
          print(f"Saved mask and overlay for {item['label']} to {output_dir}")

    # Example usage
    if __name__ == "__main__":
      extract_segmentation_masks("path/to/image.png")

Check the
[segmentation example](https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Spatial_understanding.ipynb#scrollTo=WQJTJ8wdGOKx)
in the cookbook guide for a more detailed example.
![A table with cupcakes, with the wooden and glass objects highlighted](https://ai.google.dev/static/gemini-api/docs/images/segmentation.jpg) An example segmentation output with objects and segmentation masks

## Supported image formats

Gemini supports the following image format MIME types:

- PNG - `image/png`
- JPEG - `image/jpeg`
- WEBP - `image/webp`
- HEIC - `image/heic`
- HEIF - `image/heif`

To learn about other file input methods, see the
[File input methods](https://ai.google.dev/gemini-api/docs/file-input-methods) guide.

## Capabilities

All Gemini model versions are multimodal and can be utilized in a wide range of
image processing and computer vision tasks including but not limited to image captioning,
visual question and answering, image classification, object detection and segmentation.

Gemini can reduce the need to use specialized ML models depending on your quality and performance requirements.

Some later model versions are specifically trained improve accuracy of specialized tasks in addition to generic capabilities:

- **Gemini 2.0 models** are further trained to support enhanced [object detection](https://ai.google.dev/gemini-api/docs/image-understanding#object-detection).

- **Gemini 2.5 models** are further trained to support enhanced [segmentation](https://ai.google.dev/gemini-api/docs/image-understanding#segmentation) in addition to [object detection](https://ai.google.dev/gemini-api/docs/image-understanding#object-detection).

## Limitations and key technical information

### File limit

Gemini 2.5 Pro/Flash and 2.0 Flash support a
maximum of 3,600 image files per request.

### Token calculation

- **Gemini 2.0 Flash and Gemini 2.5 Flash/Pro**: 258 tokens if both dimensions \<= 384 pixels. Larger images are tiled into 768x768 pixel tiles, each costing 258 tokens.

A rough formula for calculating the number of tiles is as follows:

- Calculate the crop unit size which is roughly: floor(min(width, height) / 1.5).
- Divide each dimension by the crop unit size and multiply together to get the number of tiles.

For example, for an image of dimensions 960x540 would have a crop unit size
of 360. Divide each dimension by 360 and the number of tile is 3 \* 2 = 6.

### Media resolution

Gemini 3 introduces granular control over multimodal vision processing with the
`media_resolution` parameter. The `media_resolution` parameter determines the
**maximum number of tokens allocated per input image or video frame.**
Higher resolutions improve the model's ability to
read fine text or identify small details, but increase token usage and latency.

For more details about the parameter and how it can impact token calculations,
see the [media resolution](https://ai.google.dev/gemini-api/docs/media-resolution) guide.

## Tips and best practices

- Verify that images are correctly rotated.
- Use clear, non-blurry images.
- When using a single image with text, place the text prompt *after* the image part in the `contents` array.

## What's next

This guide shows you how to upload image files and generate text outputs from image
inputs. To learn more, see the following resources:

- [Files API](https://ai.google.dev/gemini-api/docs/files): Learn more about uploading and managing files for use with Gemini.
- [System instructions](https://ai.google.dev/gemini-api/docs/text-generation#system-instructions): System instructions let you steer the behavior of the model based on your specific needs and use cases.
- [File prompting strategies](https://ai.google.dev/gemini-api/docs/files#prompt-guide): The Gemini API supports prompting with text, image, audio, and video data, also known as multimodal prompting.
- [Safety guidance](https://ai.google.dev/gemini-api/docs/safety-guidance): Sometimes generative AI models produce unexpected outputs, such as outputs that are inaccurate, biased, or offensive. Post-processing and human evaluation are essential to limit the risk of harm from such outputs.
