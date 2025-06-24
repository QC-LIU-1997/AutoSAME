from SAME.models.segment_anything_same.build_sam_e import same_model_registry
def get_model(modelname="SAME", args=None, opt=None):
    if modelname == "SAME":
        model = same_model_registry['vit_b'](args=args, checkpoint=args.sam_ckpt)
    else:
        raise RuntimeError("Could not find the model:", modelname)
    return model
