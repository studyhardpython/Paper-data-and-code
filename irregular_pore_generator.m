% irregular_pore_generator(48,512,512,0.4,0.15,1);
function [img, final_porosity] = irregular_pore_generator(zhijing, width, height, porosity, dR, lambda_ar)
radius = zhijing / 2;
dx = 71;
% dx: seed point column spacing (pixels), manually tuned for each
% combination of pore diameter (zhijing) and target porosity to ensure
dy = dx * sqrt(3) / 2;
start_x = radius + 5;
start_y = radius + 5;
usable_width = width - 2 * radius;
usable_height = height - 2 * radius;
n_rows = floor(usable_height / dy) + 1;
n_cols = floor(usable_width / dx) + 1;

seed_points = zeros(n_rows * n_cols, 2);
point_count = 0;
for row = 0:n_rows-1
    y = start_y + row * dy;
    if y > height - radius
        continue;
    end
    col_offset = mod(row, 2) * dx / 2;
    for col = 0:n_cols-1
        x = start_x + col * dx + col_offset;
        if x < radius || x > width - radius
            continue;
        end
        point_count = point_count + 1;
        seed_points(point_count, :) = [x, y];
    end
end
seed_points = seed_points(1:point_count, :);

img = ones(height, width);
[img_xx, img_yy] = meshgrid(1:width, 1:height);

num_points = size(seed_points, 1);
placed_max_radii = zeros(1, num_points);
placed_polygons = cell(1, num_points);
safety_margin = 5;
num_control_points = 6;
max_attempts = 20;

for i = 1:num_points
    x_center = seed_points(i, 1);
    y_center = seed_points(i, 2);
    current_radius = radius;
    attempt = 0;
    success = false;

    while ~success && attempt < max_attempts
        base_angles = linspace(0, 2*pi, num_control_points)';
        radius_offsets = 1 + dR * (2*rand(num_control_points, 1) - 1);
        control_radii = current_radius * radius_offsets;

        x_points = x_center + control_radii .* cos(base_angles);
        y_points = y_center + control_radii .* sin(base_angles);
        x_points(end) = x_points(1);
        y_points(end) = y_points(1);

        t = 1:length(x_points);
        num_interp = 200;
        ts = linspace(1, length(x_points), num_interp);
        spline_x = spline(t, x_points, ts);
        spline_y = spline(t, y_points, ts);

        poly_vertices = [spline_x', spline_y'];

        dists = sqrt((poly_vertices(:,1)-x_center).^2 + (poly_vertices(:,2)-y_center).^2);
        max_radius = max(dists);

        overlap = false;
        for j = 1:(i-1)
            d_center = sqrt((x_center - seed_points(j,1))^2 + (y_center - seed_points(j,2))^2);
            if d_center < max_radius + placed_max_radii(j) + safety_margin
                overlap = true;
                break;
            end
        end

        if ~overlap
            success = true;
            placed_max_radii(i) = max_radius;
            placed_polygons{i} = poly_vertices;
            poly_mask = poly2mask(poly_vertices(:,1), poly_vertices(:,2), height, width);
            img(poly_mask) = 0;
        else
            current_radius = current_radius * 0.95;
            attempt = attempt + 1;
        end
    end

    if ~success
        safe_radius = radius;
        for j = 1:(i-1)
            dist = sqrt((x_center - seed_points(j,1))^2 + (y_center - seed_points(j,2))^2);
            max_safe = dist - placed_max_radii(j) - safety_margin;
            if max_safe < safe_radius
                safe_radius = max_safe;
            end
        end
        safe_radius = max(safe_radius, radius * 0.8);
        placed_max_radii(i) = safe_radius;
        circle_mask = (img_xx - x_center).^2 + (img_yy - y_center).^2 <= safe_radius^2;
        img(circle_mask) = 0;
    end
end

img = [img(:, 2*radius+1:end), img(:, 1:radius*2)];
img = cat(1, img(2*radius+1:end, :), img(1:radius*2, :));

target_porosity = porosity;
current_porosity = 1 - sum(img(:)) / (height * width);

r0 = round(radius);
gap = 8;
min_radius = 0.8 * r0;
max_iterations = 5000;

search_width = width - radius;
search_height = height - radius;
search_region = false(height, width);
search_region(radius:search_height, radius:search_width) = true;

total_area = height * width;
target_void_area = target_porosity * total_area;
current_void_area = current_porosity * total_area;
area_needed = target_void_area - current_void_area;

if area_needed > 0
    D = bwdist(~img);
    r = r0;
    iteration = 0;

    while area_needed > 0 && r >= min_radius && iteration < max_iterations
        iteration = iteration + 1;
        candidate_map = (D >= (r + gap)) & search_region;

        if any(candidate_map(:))
            [rows, cols] = find(candidate_map);
            rand_idx = randi(length(rows));
            center_x = rows(rand_idx);
            center_y = cols(rand_idx);

            [xx, yy] = meshgrid(1:width, 1:height);
            circle_mask = (xx - center_y).^2 + (yy - center_x).^2 <= r^2;
            img(circle_mask) = 0;

            D = min(D, bwdist(~img));
            current_void_area = current_void_area + sum(circle_mask(:));
            area_needed = target_void_area - current_void_area;
        else
            r = max(r * 0.85, min_radius);
        end
    end
end

final_porosity = 1 - sum(sum(img)) / size(img,1) / size(img,2);

img = cat(1, img(end-2*radius+1:end, :), img(1:end-2*radius, :));
img = [img(:, end-2*radius+1:end), img(:, 1:end-2*radius)];
img = cat(1, img(end-2*radius+1:end,:), img(1:end-2*radius,:));
img = [img(:, end-2*radius+1:end), img(:, 1:end-2*radius)];
img = imresize(img, [round(height/lambda_ar), width], 'nearest');
img = repmat(img, lambda_ar, 1);



end