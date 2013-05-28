#include "shadermanager.h"
#include <inviwo/core/common/inviwoapplication.h>
#include <inviwo/core/util/vectoroperations.h>
#include <modules/opengl/openglmodule.h>
#include <modules/opengl/processorgl.h>

namespace inviwo {

ShaderManager::ShaderManager() {
    InviwoApplication::getRef().registerFileObserver(this);
    openGLInfoRef_ = NULL;
}

void ShaderManager::registerShader(Shader* shader) {
    shaders_.push_back(shader);
    startFileObservation(shader->getVertexShaderObject()->getAbsoluteFileName());
    std::vector<std::string> vertexIncludes = shader->getVertexShaderObject()->getIncludeFileNames();
    for (size_t i=0;i<vertexIncludes.size();i++)
        startFileObservation(vertexIncludes[i]);
    startFileObservation(shader->getFragmentShaderObject()->getAbsoluteFileName());
    std::vector<std::string> fragmentIncludes = shader->getFragmentShaderObject()->getIncludeFileNames();
    for (size_t i=0;i<fragmentIncludes.size();i++)
        startFileObservation(fragmentIncludes[i]);
}

void ShaderManager::unregisterShader(Shader* shader) {
    shaders_.erase(std::remove(shaders_.begin(), shaders_.end(), shader), shaders_.end());
    stopFileObservation(shader->getVertexShaderObject()->getAbsoluteFileName());
    std::vector<std::string> vertexIncludes = shader->getVertexShaderObject()->getIncludeFileNames();
    for (size_t i=0;i<vertexIncludes.size();i++)
        stopFileObservation(vertexIncludes[i]);
    stopFileObservation(shader->getFragmentShaderObject()->getAbsoluteFileName());
    std::vector<std::string> fragmentIncludes = shader->getFragmentShaderObject()->getIncludeFileNames();
    for (size_t i=0;i<fragmentIncludes.size();i++)
        stopFileObservation(fragmentIncludes[i]);
}

void ShaderManager::fileChanged(std::string shaderFilename) {
    if (dynamic_cast<BoolProperty*>(InviwoApplication::getPtr()->getSettings()->getPropertyByIdentifier("shaderReloading"))->get()) { 
        if (isObserved(shaderFilename)) {
            bool successfulReload = false;
            for (size_t i=0; i<shaders_.size(); i++) {
                bool relink = false;
                std::vector<std::string> vertexIncludes = shaders_[i]->getVertexShaderObject()->getIncludeFileNames();
                if (shaders_[i]->getVertexShaderObject()->getAbsoluteFileName()==shaderFilename ||
                    std::find(vertexIncludes.begin(), vertexIncludes.end(), shaderFilename) != vertexIncludes.end()) {
                    successfulReload = shaders_[i]->getVertexShaderObject()->rebuild();
                    relink = true;
                }
                std::vector<std::string> fragmentIncludes = shaders_[i]->getFragmentShaderObject()->getIncludeFileNames();
                if (shaders_[i]->getFragmentShaderObject()->getAbsoluteFileName()==shaderFilename ||
                    std::find(fragmentIncludes.begin(), fragmentIncludes.end(), shaderFilename) != fragmentIncludes.end()) {
                    successfulReload = shaders_[i]->getFragmentShaderObject()->rebuild();
                    relink = true;
                }
                if (relink) shaders_[i]->link();
            }
            if (successfulReload) {
                LogInfo(shaderFilename + " successfuly reloaded");
                InviwoApplication::getRef().playSound(InviwoApplication::IVW_OK);
                std::vector<Processor*> processors = InviwoApplication::getRef().getProcessorNetwork()->getProcessors();
                for (size_t i=0;i<processors.size();i++)
                    if (dynamic_cast<ProcessorGL*>(processors[i]))
                        processors[i]->invalidate();
            } else InviwoApplication::getRef().playSound(InviwoApplication::IVW_ERROR);
        }
    }
}

std::string ShaderManager::getGlobalGLSLHeader() {
    if (!openGLInfoRef_){
        OpenGLModule* openGLModule = getTypeFromVector<OpenGLModule>(InviwoApplication::getRef().getModules());
        if (openGLModule)
            openGLInfoRef_ = getTypeFromVector<OpenGLInfo>(openGLModule->getResourceInfos());
    }

    if (openGLInfoRef_){
        return openGLInfoRef_->getCurrentGlobalGLSLHeader();
    }

    return "";
}

} // namespace
