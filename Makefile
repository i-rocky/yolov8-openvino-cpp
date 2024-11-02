run:
	mkdir -p build
	cd build && cmake -DCMAKE_BUILD_TYPE=Release ..
	cd build && cmake --build .
	cd build && ./detect ../model/yolo11s-seg_openvino_model/yolo11s-seg.xml

.PHONY: run
