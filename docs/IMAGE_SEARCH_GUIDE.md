# 🖼️ Guide: Image Search System ([Back to README](../README.md))

## 1. Overview

This system provides powerful image retrieval capabilities based on OpenAI's CLIP model, supporting:
- 📤 **Image Upload & Indexing**: Store images in the system and generate semantic embeddings.
- 🔍 **Image-to-Image Search**: Find visually similar images using a query image.
- 💬 **Text-to-Image Search**: Search for matching images using a natural language description.
- 📋 **Image Management**: View, delete, and manage the image library.

## 2. Technical Architecture

### Core Components
- **`ImageService`**: The backend service for image indexing and retrieval.
- **CLIP Model**: Uses `openai/clip-vit-base-patch32` from Hugging Face.
- **Vector Storage**: Employs NumPy arrays to store high-dimensional embedding vectors for images.
- **Similarity Metric**: Uses Cosine Similarity for image matching.

### Data Flow
1.  **Upload**: Image Upload → CLIP Encoding → Vector Storage → Index Saved
2.  **Search**: Query (Image/Text) → CLIP Encoding → Similarity Calculation → Results Ranking → Matched Items Returned

## 3. Feature Details

### 1. 📤 Image Upload

**Location**: `🖼️ Image Search System` → `📤 Image Upload` Tab

**Functionality**:
- Supports common image formats (JPG, PNG, GIF, BMP, etc.).
- Automatically generates a unique ID for each image based on its MD5 hash.
- Allows adding a description and comma-separated tags.
- Provides a real-time preview of the uploaded image.

**How to Use**:
1.  Click **"Select Image File"** to upload an image.
2.  Enter a description in the **"Image Description"** box (optional).
3.  Enter tags, separated by commas, in the **"Image Tags"** box (optional).
4.  Click **"📤 Upload Image"** to complete the process.

### 2. 🔍 Image-to-Image Search

**Location**: `🖼️ Image Search System` → `🔍 Image-to-Image` Tab

**Functionality**:
- Upload a query image to find visually similar ones.
- Adjust the number of results to return (1-20).
- Displays similarity scores and image metadata.
- Provides a gallery view for quick visual inspection.

**How to Use**:
1.  Click **"Select Query Image"** to upload the image you want to search with.
2.  Adjust the **"Number of results"** slider.
3.  Click **"🔍 Search by Image"**.
4.  View the results in the table and the image gallery below.

### 3. 💬 Text-to-Image Search

**Location**: `🖼️ Image Search System` → `💬 Text-to-Image` Tab

**Functionality**:
- Enter a text description to find semantically matching images.
- Supports both English and Chinese queries.
- Leverages CLIP's cross-modal understanding.
- Displays a relevance score for each result.

**How to Use**:
1.  Enter a descriptive query in the **"Search Text"** box.
2.  Adjust the **"Number of results"** slider.
3.  Click **"💬 Search by Text"**.
4.  View the results in the table and image gallery.

**Example Queries**:
- `An orange cat sleeping on a bed`
- `a red car in the street`
- `beautiful sunset landscape`
- `a person running` (Chinese `人在跑步` also works)

### 4. 📋 Image Management

**Location**: `🖼️ Image Search System` → `📋 Image Management` Tab

**Functionality**:
- View statistics for the entire image library.
- Browse a list of all indexed images.
- Delete individual images.
- Clear the entire image library.

**Operations**:
- **📊 Refresh Stats**: View total image count, storage size, format distribution, etc.
- **🔄 Refresh List**: Update the list of all images.
- **🗑️ Delete Selected Image**: Deletes the image currently selected in the list.
- **🗑️ Clear All Images**: Permanently deletes the entire image library (use with caution).

## 4. Performance & Technical Specs

### Model Specifications
- **Model**: OpenAI CLIP ViT-B/32
- **Embedding Dimension**: 512-dimensional vectors
- **Device Support**: Auto-detects and uses GPU if available, otherwise CPU.
- **Image Resolution**: Automatically resized to 224x224 for processing.

### Storage Structure
```
models/images/
├── image_index.json      # Metadata index for images
├── image_embeddings.npy  # Matrix of image embedding vectors
└── [image_id].[ext]      # Stored image file
```

### Performance Metrics
- **Encoding Speed**: Approx. 1-3 seconds per image (on CPU).
- **Search Speed**: Millisecond-level response time.
- **Storage Efficiency**: Approx. 2 KB per image for the embedding data.
- **Similarity Precision**: Based on Cosine Similarity, score ranges from 0 to 1.

## 5. Best Practices & Tips

### Best Practices
1.  **Image Quality**: Use clear, high-quality images for better search results.
2.  **Accurate Descriptions**: Add precise descriptions and tags to improve management and future searchability.
3.  **Consistent Tags**: Use a consistent tagging system for better organization.
4.  **Regular Backups**: Periodically back up the `models/images/` directory.

### Search Tips
1.  **Text-to-Image**:
    - Use specific, descriptive words.
    - Include visual characteristics like colors, shapes, and actions.
    - Both English and Chinese are supported, though English may yield better results with the base CLIP model.
2.  **Image-to-Image**:
    - The query image should clearly feature the main visual elements you are looking for.
    - The composition and angle of the query image can influence results.

### Important Notes
- The first time the system starts, it will download the CLIP model (approx. 1 GB).
- Performance is significantly better in a GPU environment.
- Uploaded images are copied to the system's storage directory, so be mindful of disk space.
- Deletion operations are irreversible.

## 6. Troubleshooting

### Common Issues
1.  **Model Fails to Load**: Check your internet connection. The model needs to be downloaded on first use.
2.  **Image Upload Fails**: Verify the image format and file size.
3.  **No Search Results**: Ensure images have been indexed in the library. Try different queries.
4.  **Slow Performance**: Consider using a GPU or reducing the number of results to return.

The system includes error handling, and any issues will be reported in the corresponding status boxes. Check the console output for more detailed error messages.
