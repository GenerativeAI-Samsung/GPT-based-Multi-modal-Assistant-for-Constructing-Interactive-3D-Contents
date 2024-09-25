
# ------------------------------------------------------------------------------
# Split in here

# Phase 2: Import các vật thể từ trong database 3D object
    assets = {}
    for object in object_list:
        assets[object["name"]] = None
        import_obj = f'assets["{object["name"]}"] = import_object("/home/khai/Desktop/Repo/GPT-based-Multi-modal-Assistant-for-Constructing-Interactive-3D-Contents/object/{object["name"]}.obj", location={initial_position[object["name"]]}, orientation=(1.5708, 0, 0))'
        exec(import_obj)        

# Phase 3.2.3: Thực hiện khởi tạo môi trường
    best_layout, best_score = constraint_solving(assets=assets, constraints=constraints)

    for asset in best_layout.items():
        name, object = asset
        print(object.location)
        # Set location
        object.imported_object.location = object.location

        # Set orientation (convert Euler angles to radians)
        orientation_rad = tuple(angle for angle in object.orientation)
        object.imported_object.rotation_euler = orientation_rad

    bpy.context.view_layer.update()
