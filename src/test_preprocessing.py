from utils.preprocessing import attempt_to_clean, is_text_clean

test_samples = [
    ' Harus gasket Anda. ###> Must be your gaskets. ',
    'Art? ###> Art? ',
    ' Kebanyakan dari mereka budak diberikan kepada mereka sebagai hadiah. ###> Most of their slaves were given to them as gifts.',
    'Naikkan mereka! ###> Let\'s go, get them on!',
    'Ini.. ini sangat aneh. ###> This is... this is just too weird.',
    'Mungkin dendam rahasia. ###> Never. Must be a secret grudge.',
    'Bagaimana kabarmu, baik? ###> Hey, Peter.',
    'Detektif Lee Tae Kyun. ###> Detective Lee Tae Kyun.',
    'aku tidak akan mengatakan apapun padanya. ###> And I won\'t tell her anything before that.'
]

invalid_test_samples = [
    'Invalid! Sentence. ###> This should fail.',
    'Ganjil? ###> Odd? ',
    'This is a test.###> No space around separator', 
    '###> No Indonesian part.',  
    'Missing English part ###> ', 
    'Valid Indonesian ###> Invalid English!!!', 
    'Too many spaces     ###> Proper sentence',
    'Bagaimana  kabarmu? ###> How are you? ',
    'Invalid chars: @#$%^&*() ###> Must be a test.', 
    '123456 ###> 123456',
]

if __name__ == "__main__":
    
    for s in test_samples:
        s = attempt_to_clean(s)
        print(s)
        print(is_text_clean(s))
        print()

    print()

    for s in invalid_test_samples:
        s = attempt_to_clean(s)
        print(s)
        print(is_text_clean(s))
        print()