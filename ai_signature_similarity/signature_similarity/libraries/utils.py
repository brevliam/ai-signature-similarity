from ..models import Signature

def find_saved_signatures_by_nik(nik):
    anchor_sign = Signature.objects.get(nik=nik, is_anchor=True)
    test_sign = Signature.objects.get(nik=nik, is_anchor=False)
    anchor_path = anchor_sign.img.path
    test_path = test_sign.img.path
    
    return anchor_path, test_path

def delete_signature_data_by_nik(nik):
    Signature.objects.filter(nik=nik).delete()