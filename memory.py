 for i in range(len(faces)):
            dd=top_exp.number_of_edges_from_face(faces[i])
            edgess=top_exp.edges_from_face(faces[i])
            edges=[edge for edge in edgess]
            
            for j in range(len(edges)):
                face_test_all=top_exp.faces_from_edge(edges[j])
                face_test_all_list=[face for face in face_test_all]
                if(len(face_test_all_list)>1):
                    for k in range(len(face_test_all_list)):
                        for m in range(len(face_test_all_list)):
                            if k==m:continue 
                            if(face_test_all_list[k].IsEqual(face_test_all_list[m])):
                                a=1











