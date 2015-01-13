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

#ifndef IVW_VOLUMESLICEGL_H
#define IVW_VOLUMESLICEGL_H

#include <modules/basegl/baseglmoduledefine.h>

#include <inviwo/core/common/inviwo.h>
#include <inviwo/core/datastructures/geometry/geometrytype.h>
#include <inviwo/core/ports/imageport.h>
#include <inviwo/core/ports/volumeport.h>
#include <inviwo/core/processors/processor.h>
#include <inviwo/core/properties/baseoptionproperty.h>
#include <inviwo/core/properties/boolproperty.h>
#include <inviwo/core/properties/compositeproperty.h>
#include <inviwo/core/properties/ordinalproperty.h>
#include <inviwo/core/properties/transferfunctionproperty.h>
#include <inviwo/core/properties/eventproperty.h>
#include <modules/opengl/inviwoopengl.h>

namespace inviwo {

class Shader;
class Mesh;

class IVW_MODULE_BASEGL_API VolumeSliceGL : public Processor {
public:
    VolumeSliceGL();
    ~VolumeSliceGL();

    InviwoProcessorInfo();

    void initialize();
    void deinitialize();

    virtual void invokeInteractionEvent(Event*);

    bool positionModeEnabled() const { return posPicking_.get(); }

protected:
    void process();

    void shiftSlice(int);

    void sliceAxisChanged();
    void planeSettingsChanged();
    void tfMappingEnabledChanged();
    void updateMaxSliceNumber();
    void posPickingChanged();

    void renderPositionIndicator();
    void updateIndicatorMesh();
    void updateReadOnlyStates();

    // updates the selected position, pos is given in normalized viewport coordinates, i.e. [0,1]
    void setVolPosFromScreenPos(vec2 pos);
    vec2 getScreenPosFromVolPos();

    void invalidateMesh();
    int getSliceNumber() const;
    float getNormalizedSliceNumber() const;

    // getAxisIndex() is used to decouple potential changes in CoordinateEnums from indexing
    inline int getAxisIndex(int axis) const {
        // axis should be of type CoordinateEnums::CartesianCoordinateAxis
        switch (axis) {
            case inviwo::CoordinateEnums::X:
                return 0;
            case inviwo::CoordinateEnums::Y:
                return 1;
            case inviwo::CoordinateEnums::Z:
                return 2;
            default:
                return 0;
        }
    }

private:
    void updatePos();

    void eventShiftSlice(Event*);
    void eventSetMarker(Event*);
    void eventStepSliceUp(Event*);
    void eventStepSliceDown(Event*);
    void eventGestureShiftSlice(Event*);

    VolumeInport inport_;
    ImageOutport outport_;

    CompositeProperty trafoGroup_;
    CompositeProperty pickGroup_;
    CompositeProperty tfGroup_;

    OptionPropertyInt sliceAlongAxis_;  // Axis enum
    IntProperty sliceX_;
    IntProperty sliceY_;
    IntProperty sliceZ_;

    FloatVec3Property worldPosition_;

    FloatVec3Property planeNormal_;
    FloatProperty planeOffset_;
    IntProperty* slices_[4];                  // array to access the individual slices via index
    OptionPropertyFloat rotationAroundAxis_;  // Clockwise rotation around slice axis
    BoolProperty flipHorizontal_;
    BoolProperty flipVertical_;
    OptionPropertyInt volumeWrapping_; 

    BoolProperty posPicking_;
    BoolProperty showIndicator_;
    FloatVec4Property indicatorColor_;

    BoolProperty tfMappingEnabled_;
    TransferFunctionProperty transferFunction_;
    FloatProperty tfAlphaOffset_;

    BoolProperty handleInteractionEvents_;

    EventProperty mouseShiftSlice_;
    EventProperty mouseSetMarker_;

    EventProperty stepSliceUp_;
    EventProperty stepSliceDown_;

    EventProperty gestureShiftSlice_;

    Shader* shader_;
    Shader* indicatorShader_;

    Mesh* meshCrossHair_;
    Mesh* meshBox_;  // second mesh needed since Mesh does not support multiple connectivity types
    bool meshDirty_;

    uvec3 volumeDimensions_;
};
}

#endif  // IVW_VOLUMESLICEGL_H
