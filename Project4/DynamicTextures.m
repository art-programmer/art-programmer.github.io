function [ extended_video ] = DynamicTextures( video, num_new_frames, num_components, num_previous_frames )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

[image_height, image_width, num_channels, num_frames] = size(video);
image_vec = reshape(double(video), image_width * image_height * num_channels, num_frames);

image_mean = mean(image_vec, 2);
image_vec = image_vec - image_mean * ones(1, num_frames);

coefficients = pca(image_vec', 'NumComponents', num_components);

projections = coefficients' * image_vec;

reconstructed_image_vec = coefficients * projections;
reconstruction_error = reconstructed_image_vec - image_vec;
reconstruction_svar = sqrt(sum(reconstruction_error.^2, 2) / num_frames);

A = zeros((num_frames - num_previous_frames) * num_components, num_previous_frames * num_components^2);
b = zeros((num_frames - num_previous_frames) * num_components, 1);
for frame = num_previous_frames + 1:num_frames
    for component = 1:num_components
        for previous_frame = frame - num_previous_frames:frame - 1
            A((frame - num_previous_frames - 1) * num_components + component, (component - 1) * num_previous_frames * num_components + (previous_frame - (frame - num_previous_frames)) * num_components + 1 : (component - 1) * num_previous_frames * num_components + (previous_frame - (frame - num_previous_frames)) * num_components + num_components) = projections(:, previous_frame)';
        end
        b((frame - num_previous_frames - 1) * num_components + component) = projections(component, frame);
    end
end
transition = linsolve(A, b);
transition = reshape(transition, num_previous_frames * num_components, num_components)';

transition_errors = zeros(num_components, num_frames - num_previous_frames);
for frame = num_previous_frames + 1:num_frames
    previous_projections = reshape(projections(:, frame - num_previous_frames:frame - 1), [num_previous_frames * num_components, 1]);
    transition_errors(:, frame - num_previous_frames) = transition * previous_projections - projections(:, frame);
end
transition_svar = sqrt(sum(transition_errors.^2, 2) / (num_frames - num_previous_frames));

extended_projections = horzcat(projections, zeros(num_components, num_new_frames));
for frame = num_frames + 1:num_frames + num_new_frames
    previous_projections = reshape(extended_projections(:, frame - num_previous_frames:frame - 1), [num_previous_frames * num_components, 1]);
    extended_projections(:, frame) = transition * previous_projections + randn(num_components, 1) .* transition_svar * 0;
end

% extended_projections = randn(num_components, num_new_frames);
% for frame = num_previous_frames + 1:num_frames + num_new_frames
%     previous_projections = reshape(extended_projections(:, frame - num_previous_frames:frame - 1), [num_previous_frames * num_components, 1]);
%     extended_projections(:, frame) = transition * previous_projections + randn(num_components, 1) .* transition_svar;
% end

extended_image_vec = coefficients * extended_projections + image_mean * ones(1, num_frames + num_new_frames);
%extended_image_vec(:, num_frames + 1:num_frames + num_new_frames) = extended_image_vec(:, num_frames + 1:num_frames + num_new_frames) + (randn() * reconstruction_svar) * ones(1, num_new_frames);
extended_image_vec(:, num_frames + 1:num_frames + num_new_frames) = extended_image_vec(:, num_frames + 1:num_frames + num_new_frames) + randn(image_height * image_width * num_channels, num_new_frames) .* (reconstruction_svar * ones(1, num_new_frames)) * 0;
extended_image_vec(find(extended_image_vec > 255)) = 255;
extended_image_vec(find(extended_image_vec < 0)) = 0;

extended_video = uint8(reshape(extended_image_vec, [image_height, image_width, num_channels, num_frames + num_new_frames]));

end

