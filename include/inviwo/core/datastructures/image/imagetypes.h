/*********************************************************************************
 *
 * Inviwo - Interactive Visualization Workshop
 * Version 0.6b
 *
 * Copyright (c) 2013-2015 Inviwo Foundation
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 *********************************************************************************/

#ifndef IVW_IMAGETYPES_H
#define IVW_IMAGETYPES_H

#include <inviwo/core/common/inviwocoredefine.h>
#include <inviwo/core/common/inviwo.h>
#include <inviwo/core/io/serialization/ivwdeserializer.h>

namespace inviwo {

enum ImageType {
    COLOR_ONLY = 0,
    COLOR_DEPTH = 1,
    COLOR_PICKING = 2,
    COLOR_DEPTH_PICKING = 3
};

enum LayerType {
    COLOR_LAYER = 0,
    DEPTH_LAYER = 1,
    PICKING_LAYER = 2
};

STARTCLANGIGNORE("-Wunused-function")
static bool typeContainsColor(ImageType type) {
    return (type == COLOR_ONLY || type == COLOR_DEPTH || type == COLOR_PICKING || type == COLOR_DEPTH_PICKING);
}

static bool typeContainsDepth(ImageType type) {
    return (type == COLOR_DEPTH || type == COLOR_DEPTH_PICKING);
}

static bool typeContainsPicking(ImageType type) {
    return (type == COLOR_PICKING || type == COLOR_DEPTH_PICKING);
}
ENDCLANGIGNORE
} // namespace

#endif // IVW_IMAGETYPES_H
