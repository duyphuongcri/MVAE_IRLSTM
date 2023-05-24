path_root = "D:\0.PhD\Dataset\ADNI\ADNI\"
listFiles =dir(fullfile(path_root, '**', '*.hdr'))

for i = 1:940
    path_hdr = listFiles(i).folder + "\" + listFiles(i).name;
    path_img = listFiles(i).folder + "\" + strrep(listFiles(i).name,'hdr','img');
    path_nii = listFiles(i).folder + "\" + strrep(listFiles(i).name,'hdr','nii');
    info = niftiinfo(path_hdr);
    img = niftiread(path_img);
    nii_new = permute(img, [1, 2, 3]);
    niftiwrite(nii_new, path_nii, info);
    disp(i);
    disp(path_nii);
end
