/*********************************************************************************
 *
 * Inviwo - Interactive Visualization Workshop
 *
 * Copyright (c) 2014-2015 Inviwo Foundation
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

#ifndef IVW_IMAGEOVERLAYGL_H
#define IVW_IMAGEOVERLAYGL_H

#include <modules/basegl/baseglmoduledefine.h>
#include <inviwo/core/common/inviwo.h>
#include <inviwo/core/processors/processor.h>
#include <inviwo/core/interaction/events/mouseevent.h>
#include <inviwo/core/properties/boolproperty.h>
#include <inviwo/core/properties/baseoptionproperty.h>
#include <inviwo/core/properties/ordinalproperty.h>
#include <inviwo/core/interaction/interactionhandler.h>
#include <inviwo/core/ports/imageport.h>
#include <inviwo/core/ports/multidatainport.h>
#include <modules/opengl/glwrap/shader.h>
#include <modules/opengl/inviwoopengl.h>

namespace inviwo {

class Shader;

/*! \class OverlayProperty
*
* \brief CompositeProperty for overlay images. An overlay is defined by
* a position, a anchor position, and the blend mode used for compositing.
*/
// TODO: implement absolute positioning.
//    this will require the image dimensions of the source and the overlay
class IVW_MODULE_BASEGL_API OverlayProperty : public CompositeProperty {
public:
    enum class Positioning {
        Relative,
        Absolute,
    };
    enum class BlendMode {
        Replace,
        Over,
    };

    OverlayProperty(std::string identifier,
        std::string displayName,
        InvalidationLevel invalidationLevel=INVALID_OUTPUT,
        PropertySemantics semantics=PropertySemantics::Default);
    virtual ~OverlayProperty() {}

    void updateViewport(vec2 destDim);

    ivec4 viewport_;

    FloatVec2Property size_;
    FloatVec2Property pos_;
    FloatVec2Property anchorPos_;

    TemplateOptionProperty<BlendMode> blendMode_;

    // TODO: consider absolute positioning
    //TemplateOptionProperty<Positioning> positioning_;
    //TemplateOptionProperty<Positioning> anchorPositioning_;
};


/** \docpage{org.inviwo.ImageOverlayGL, Image Overlay}
* Places one or more input images on top of the source image.
* ![](imageoverlay.png)
*
* ### Inports
*   * __ImageInport__ Source image.
*   * __ImageInport__ Overlay images (multi-port).
*
* ### Outports
*   * __ImageOutport__ The output image.
*
* ### Properties
*   * __Overlay Interaction__ Allow interactions on overlay images.
*   * __Size__              Size of the overlay image.
*   * __Position__          Position of the overlay image.
*   * __Anchor Position__   Anchor of the overlay image for alignment.
*   * __Blend Mode__        Blend mode used for mixing the overlay image.
*/

/*! \class ImageOverlayGL
*
* \brief Places one or more input images on top of the source image.
*/
class IVW_MODULE_BASEGL_API ImageOverlayGL : public Processor {
public:
    ImageOverlayGL();
    ~ImageOverlayGL();

    InviwoProcessorInfo();

    virtual void initialize() override;
    virtual void deinitialize() override;

    const std::vector<Inport*>& getInports(Event*) const;
    const std::vector<ivec4>& getViewCoords() const;

    bool isReady() const override;

protected:
    virtual void process() override;

    void overlayInportChanged();
    void updateViewports(bool force = false);
    void updateDimensions();
    void onStatusChange();
    void onOverlayInteractionChange();

    class ImageOverlayGLInteractionHandler : public InteractionHandler {
    public:
        ImageOverlayGLInteractionHandler(ImageOverlayGL*);
        ~ImageOverlayGLInteractionHandler(){};

        void invokeEvent(Event* event);
        ivec2 getActivePosition() const { return activePosition_; }

    private:
        ImageOverlayGL* src_;
        MouseEvent activePositionChangeEvent_;
        bool viewportActive_;
        ivec2 activePosition_;
    };

private:
    static bool inView(const ivec4& view, const ivec2& pos);
    ImageInport inport_;
    ImageInport overlayPort_;
    ImageOutport outport_;
    
    BoolProperty overlayInteraction_; //<! allows to interact with overlay images, otherwise only the source image will receive interaction events

    // TODO: replace this with std::vector to match multi-inport
    OverlayProperty overlayProperty_;

    Shader shader_;

    ImageOverlayGLInteractionHandler overlayHandler_;

    ivec2 currentDim_;

    std::vector<ivec4> viewCoords_;

    mutable std::vector<Inport*> currentInteractionInport_;
};

}  // namespace

#endif  // IVW_IMAGEOVERLAYGL_H