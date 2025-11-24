from models.s1_records.model_record import ModelRecordLlamaCpp, ModelRecordLMStudio


ModelRecordLocalAny = ModelRecordLlamaCpp
ModelRecordRemoteAny = ModelRecordLMStudio
ModelRecordAny = ModelRecordLocalAny | ModelRecordRemoteAny
