%% VMD 分解 - 处理 A 类所有音频 + 计时（纯基础分解）
%清空工作区所有变量；清空命令行窗口；关掉所有图窗
clear; clc; close all;              

%% =====================  路径配置  =====================
rootPath = 'D:\Work\pythonproject\Datasets\Data_Sets\Switch_machine_audio\data';      % 数据集地址
savePath = 'D:\Work\pythonproject\fault_switch\VMD_fault\outputs\base_VMD';           % 文件保存地址
%% ===================== 固定配置 =====================
dataSets = {'train','test'}; % 固定两个子集
category = {'A','B','C','D','E','F','G','H','I','J'}; % 10个类别
classPath = fullfile(rootPath, targetClass);
NumIMF = 6;                 % VMD 分解模态数（可改）

%% ===================== 自动遍历：子集→类别→所有音频 =====================
for d = 1:length(dataSets)
    % 当前子集：train / test
    setName = dataSets{d};
    setPath = fullfile(rootPath, setName);

    % 遍历 A-J 每个类别
    for c = 1:length(category)
        className = category{c};
        classPath = fullfile(setPath, className);

        % 如果文件夹不存在，跳过
        if ~exist(classPath,'dir')
            fprintf('→ 跳过：%s\n', classPath);
            continue;
        end

        % 读取该文件夹下所有 wav 音频
        audioList = dir(fullfile(classPath, '*.wav'));
        if isempty(audioList)
            fprintf('→ %s-%s 无音频文件\n', setName, className);
            continue;
        end

        fprintf('\n=====================================================\n');
        fprintf('正在处理：【%s】→【%s】，共 %d 个音频\n', setName, className, length(audioList));
        fprintf('=====================================================\n');

        % 遍历当前类别下所有音频
        for i = 1:length(audioList)
            % 1. 获取音频路径
            fileName = audioList(i).name;
            filePath = fullfile(classPath, fileName);

            fprintf('\n-----------------------------------------------------\n');
            fprintf('处理 %s-%s：%d/%d → %s\n', setName, className, i, length(audioList), fileName);

            % 2. 读取音频
            [x, fs] = audioread(filePath);
            x = x(:, 1); % 单声道

            % 3. VMD 分解 + 计时
            tic;
            [u, ~, ~] = VMD(x, 2500, 1e-7, NumIMF, 0, 1, 1);
            time_cost = toc;
            fprintf('VMD分解耗时：%.4f 秒\n', time_cost);

            % ===================== 核心：自动复刻目录结构保存 =====================
            % 自动生成和源文件完全一样的保存路径
            savePath = fullfile(saveRoot, setName, className);
            if ~exist(savePath,'dir')
                mkdir(savePath); % 自动创建文件夹
            end

            % 保存文件名和源音频完全一致（仅后缀改为_vmd.mat）
            [~, nameOnly, ~] = fileparts(fileName);
            saveFile = fullfile(savePath, [nameOnly '_vmd.mat']);

            % 保存分解结果
            save(saveFile, 'x', 'u', 'fs', 'time_cost');
            fprintf('✅ 已保存：%s\n', saveFile);
        end
    end
end

fprintf('\n\n=====================================================\n');
fprintf('🎉 所有文件 VMD 分解 + 保存全部完成！\n');
fprintf('=====================================================\n');