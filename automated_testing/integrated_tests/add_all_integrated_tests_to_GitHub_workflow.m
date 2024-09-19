function add_all_integrated_tests_to_GitHub_workflow
% Add all the integrated tests to the GitHub Workflow file so they will be
% run automatically
%
% NOTE: this script must be run from inside automated_testing/integrated_tests!

foldername_integrated_tests = pwd;
foldername_workflows = '../../.github/workflows';

list_of_tests = list_all_integrated_tests( {}, foldername_integrated_tests);

add_tests_to_main_workflow_file( list_of_tests)

for i = 1: length( list_of_tests)
  create_single_test_workflow_file( list_of_tests{ i});
end

  function list_of_tests = list_all_integrated_tests( list_of_tests, test_path)

    henk = dir( test_path);

    is_test = false;
    for i = 1: length( henk)
      if strcmpi( henk( i).name,'config.cfg')
        is_test = true;
      end
    end
    if is_test
      list_of_tests{ end+1} = test_path( length( foldername_integrated_tests)+2:end);
      return
    end

    for i = 1: length( henk)
      if strcmpi( henk( i).name,'.') || strcmpi( henk( i).name, '..')
        continue
      end
      if henk( i).isdir
        list_of_tests = list_all_integrated_tests( list_of_tests, [test_path '/' henk( i).name]);
      end
    end

  end

  function add_tests_to_main_workflow_file( list_of_tests)

    filename_workflow_main  = [foldername_workflows ...
      '/UFE_test_suite_run_integrated_tests.yml'];

    fid = fopen( filename_workflow_main,'w');

    fprintf( fid, '%s\n', 'name: UFEMISM Test Suite - run integrated tests');
    fprintf( fid, '%s\n', 'run-name: ${{ github.actor }} - UFEMISM Test Suite - run integrated tests');
    fprintf( fid, '%s\n', 'on:');
    fprintf( fid, '%s\n', '  workflow_call:');
    fprintf( fid, '%s\n', '');
    fprintf( fid, '%s\n', 'jobs:');

    for i = 1: length( list_of_tests)

      test_path = list_of_tests{ i};
      test_name = strrep( test_path,'/','_');

      fprintf( fid, '%s\n', '');
      fprintf( fid, '%s\n', ['  ' test_name ':']);
      fprintf( fid, '%s\n', ['    uses: ./.github/workflows/zz_integrated_test_' test_name '.yml']);

    end

    fclose( fid);

  end

  function create_single_test_workflow_file( test_path)

    ii = strfind( test_path,'/'); ii = ii( end);
    test_path_firstname = test_path( ii+1:end);

    test_name           = strrep( test_path          ,'/','_');
    test_name_firstname = strrep( test_path_firstname,'/','_');

    % Read dummy workflow file
    filename_workflow_dummy = 'integrated_test_single_workflow_dummy.txt';
    fid = fopen( filename_workflow_dummy,'r');
    temp = textscan( fid,'%s','delimiter','\n','whitespace',''); temp = temp{1};
    fclose( fid);

    % Place test name in file
    for ii = 1: length( temp)
      temp{ii} = strrep( temp{ii}, '!!test_path!!'          , test_path);
      temp{ii} = strrep( temp{ii}, '!!test_name!!'          , test_name);
      temp{ii} = strrep( temp{ii}, '!!test_name_firstname!!', test_name_firstname);
    end

    % Write to single test workflow file
    filename_workflow = [foldername_workflows '/zz_integrated_test_' test_name '.yml'];
    fid = fopen( filename_workflow,'w');
    for ii = 1: length( temp)
      fprintf( fid,'%s\n',temp{ii});
    end
    fclose( fid);

  end

end