"""
After extracting the RAR, we run this to move all the files into
the appropriate train/test folders.
Should only run this file once!
"""
import os
import os.path

def get_train_test_lists(version='01'):
    """
    Using one of the train/test files (01, 02, or 03), get the filename
    breakdowns we'll later use to move everything.
    """
    # Get our files based on version.
    test_file = os.path.join('ucfTrainTestlist', 'testlist' + version + '.txt')
    train_file = os.path.join('ucfTrainTestlist', 'trainlist' + version + '.txt')

    # Build the test list.
    with open(test_file) as fin:
        test_list = ['UCF101/'+row.strip() for row in list(fin)]

    # Build the train list. Extra step to remove the class index.
    with open(train_file) as fin:
        train_list = ['UCF101/'+row.split()[0] for row in list(fin)]

    # Set the groups in a dictionary.
    file_groups = {
        'UCF101_train': train_list,
        'UCF101_test': test_list
    }

    return file_groups

def move_files(file_groups):
    """This assumes all of our files are currently in _this_ directory.
    So move them to the appropriate spot. Only needs to happen once.
    """
    # Do each of our groups.
    for group, videos in file_groups.items():

        # Do each of our videos.
        for video in videos:
            video = video.replace("\\", "/")
            # Check if the video exists.
            if not os.path.exists(video):
                print("Can't find %s. Skipping." % (video))
                continue 
            # Get the parts.
            parts = video.strip().split('/')
            datasetname = parts[0]
            classname = parts[1]
            filename = parts[2]

            # Check if this class exists in UCF101_train/test
            if not os.path.exists(os.path.join(group, classname)):
                print("Creating folder for %s/%s" % (group, classname))
                os.makedirs(os.path.join(group, classname))

            # Check if we have already moved this file, or at least that it
            # exists to move.
            if not os.path.exists(os.path.join('UCF101',classname,filename)):
                print("Can't find %s to move. Skipping." % (filename))
                continue

            # Move it.
            dest = os.path.join(group, classname, filename)
            print("Moving %s to %s" % (os.path.join('UCF101',filename), dest))
            os.rename(os.path.join('UCF101',classname,filename), dest)

    print("Done.")

def main():
    """
    Go through each of our train/test text files and move the videos
    to the right place.
    """
    # Get the videos in groups so we can move them.
    group_lists = get_train_test_lists()

    # Move the files.
    move_files(group_lists)

if __name__ == '__main__':
    main()
